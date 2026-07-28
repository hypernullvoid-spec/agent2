"""
Phase 14: LLM Fine-Tuning (LoRA / QLoRA)

Adds the ability to fine-tune a small, local, open-weight model on a
domain-specific dataset using LoRA/QLoRA via HuggingFace's PEFT
library — per the original blueprint, "useful once your agent itself
needs a cheaper, specialized model for repetitive subtasks." This is
NOT about fine-tuning Claude (Claude is accessed via the Anthropic API
and isn't something this project trains) — it's about producing a
small local model the agent's own pipeline could call for narrow,
repetitive jobs (e.g. classifying support tickets into categories,
extracting structured fields from a fixed document format) where
spinning up a full Claude call for every instance would be needlessly
expensive once the pattern is well-established.

Why LoRA/QLoRA, not full fine-tuning
─────────────────────────────────────────
Full fine-tuning updates every weight in the base model — expensive in
both compute and storage (a full copy of the model per fine-tune). LoRA
("Low-Rank Adaptation") freezes the base model and trains a small set
of low-rank adapter matrices instead — typically <1% of the base
model's parameter count. QLoRA adds 4-bit quantization of the frozen
base weights on top of that, cutting GPU memory requirements further.
Both produce a small "adapter" artifact (a few MB to a few hundred MB,
not gigabytes) that gets loaded on top of the (unmodified) base model at
inference time — which is also why disconnect/reconnect-style adapter
swapping is cheap: you're not re-downloading or re-copying the base
model, just swapping a small adapter file.

Why this module is structured the way it is
─────────────────────────────────────────────────
Three genuinely separate concerns, kept as three separate methods so
each can be tested/reasoned about independently:
  1. prepare_dataset()  — turn raw examples into the exact prompt/
     completion text format the chosen base model's chat template
     expects. Validates shape and reports a clear error string for
     malformed input, rather than letting a training run fail 20
     minutes in on bad data.
  2. fine_tune()         — build the LoRA config, load the (quantized,
     for QLoRA) base model, and run a small number of training epochs.
     This is the one step that genuinely requires a GPU and real compute
     time — everything around it is structured so failures here are
     reported the same "error as a string" way as everywhere else, not
     left to crash the whole agent process.
  3. merge_and_export()  — fold the trained LoRA adapter back into a
     standalone model directory (or leave it as a separate adapter,
     depending on export_mode), so the result can be loaded normally by
     anything expecting a regular HuggingFace model directory.

What this module does NOT do
────────────────────────────────
  - It does not pick a base model for you. base_model_id is required —
    there's no universally "right" small model, and guessing one would
    be a worse default than asking. A reasonable starting point for a
    CPU-friendly experiment is something in the 0.5B–2B parameter range
    (e.g. a small Qwen or Llama variant); GPU-backed environments can go
    larger.
  - It does not manage GPU drivers/CUDA setup. If torch.cuda isn't
    available, fine_tune() reports that plainly up front and lets the
    agent decide whether to proceed on CPU (slow, fine for a tiny
    smoke-test run) or stop.
  - It does not serve the fine-tuned model. That's a deployment concern
    (conceptually adjacent to Phase 10, though Phase 10 specifically
    packages scikit-learn-style ML models, not causal LMs) — this module
    stops at producing a usable model/adapter directory on disk.
"""

import json
import os
from pathlib import Path
from typing import Optional

WORKSPACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace")
)
FINETUNE_SUBDIR = "finetune"   # workspace/finetune/<run_id>/ holds everything for one run

DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]   # attention projections — the standard, minimal LoRA target set


class FineTuner:
    """
    Stateful only in the sense of tracking what's been fine-tuned this
    session (self._runs), so list_finetune_runs() and later phases can
    look up a completed run's adapter path without re-deriving it.
    Otherwise stateless — every method takes the inputs it needs
    explicitly, the same "no hidden state" discipline as
    FeatureEngine/ModelTrainer from Phases 7/8.
    """

    def __init__(self):
        self._runs: dict[str, dict] = {}   # run_id -> {"status", "adapter_path", "base_model_id", ...}

    # ───────────────────────────────────────────────── 1. dataset preparation

    def prepare_dataset(self, examples: list[dict], run_id: str, validation_split: float = 0.1) -> str:
        """
        Validate and format a list of {"prompt": ..., "completion": ...}
        examples into train/validation JSONL files under
        workspace/finetune/<run_id>/. Returns a clear error string (not
        a raised exception) for any malformed example, naming which
        example index and field is the problem — a fine-tuning run is
        expensive enough in time that catching bad data before training
        starts, rather than after a 20-minute run fails on row 4,000, is
        worth a dedicated validation pass.
        """
        if not examples:
            return "Error: no examples provided. Pass a non-empty list of {'prompt', 'completion'} dicts."

        for i, ex in enumerate(examples):
            if not isinstance(ex, dict):
                return f"Error: example {i} is not a dict (got {type(ex).__name__})."
            if "prompt" not in ex or "completion" not in ex:
                return f"Error: example {i} is missing 'prompt' and/or 'completion'. Got keys: {list(ex.keys())}"
            if not isinstance(ex["prompt"], str) or not ex["prompt"].strip():
                return f"Error: example {i}'s 'prompt' must be a non-empty string."
            if not isinstance(ex["completion"], str) or not ex["completion"].strip():
                return f"Error: example {i}'s 'completion' must be a non-empty string."

        if not (0.0 <= validation_split < 1.0):
            return f"Error: validation_split must be in [0.0, 1.0), got {validation_split}."

        n_val = max(1, int(len(examples) * validation_split)) if validation_split > 0 and len(examples) >= 10 else 0
        val_examples = examples[:n_val] if n_val else []
        train_examples = examples[n_val:] if n_val else examples

        run_dir = self._run_dir(run_id)
        os.makedirs(run_dir, exist_ok=True)

        train_path = os.path.join(run_dir, "train.jsonl")
        with open(train_path, "w", encoding="utf-8") as f:
            for ex in train_examples:
                f.write(json.dumps({"prompt": ex["prompt"], "completion": ex["completion"]}) + "\n")

        val_path = None
        if val_examples:
            val_path = os.path.join(run_dir, "val.jsonl")
            with open(val_path, "w", encoding="utf-8") as f:
                for ex in val_examples:
                    f.write(json.dumps({"prompt": ex["prompt"], "completion": ex["completion"]}) + "\n")

        self._runs[run_id] = {
            "status": "dataset_prepared",
            "train_path": train_path,
            "val_path": val_path,
            "n_train": len(train_examples),
            "n_val": len(val_examples),
        }

        return (
            f"Prepared dataset for run '{run_id}': {len(train_examples)} training examples"
            + (f", {len(val_examples)} validation examples" if val_examples else " (no validation split — fewer than 10 examples or validation_split=0)")
            + f".\nSaved to workspace/{FINETUNE_SUBDIR}/{run_id}/train.jsonl"
            + (f" and val.jsonl" if val_examples else "")
            + ".\nNext: call fine_tune(run_id, base_model_id=...) to start training."
        )

    # ───────────────────────────────────────────────── 2. fine-tuning

    def fine_tune(
        self,
        run_id: str,
        base_model_id: str,
        use_qlora: bool = False,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        target_modules: Optional[list[str]] = None,
    ) -> str:
        """
        Load base_model_id (quantized to 4-bit if use_qlora=True), attach
        a LoRA adapter, and train on the dataset prepared by
        prepare_dataset(run_id). Saves the trained adapter under
        workspace/finetune/<run_id>/adapter/.

        use_qlora requires bitsandbytes and a CUDA GPU — if either is
        missing, this reports that plainly rather than silently falling
        back to plain LoRA (the memory/compute tradeoff between the two
        is something the caller should decide on explicitly, not have
        silently changed under them).
        """
        run = self._runs.get(run_id)
        if run is None or run.get("status") != "dataset_prepared":
            return (
                f"Error: run '{run_id}' has no prepared dataset. "
                f"Call prepare_dataset(examples, run_id='{run_id}') first."
            )

        try:
            import torch
            from datasets import load_dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            return (
                f"Error: fine-tuning requires 'pip install torch transformers datasets peft accelerate "
                f"bitsandbytes' (bitsandbytes only needed for use_qlora=True). Missing: {e}"
            )

        cuda_available = torch.cuda.is_available()
        if use_qlora and not cuda_available:
            return (
                "Error: use_qlora=True requires a CUDA GPU (4-bit quantization via "
                "bitsandbytes is GPU-only). Either run on a GPU-backed environment, "
                "or set use_qlora=False to run plain LoRA (much slower on CPU, but "
                "will run for a small smoke-test dataset)."
            )
        if not cuda_available:
            print(
                "[finetune] No CUDA GPU detected — training will run on CPU. "
                "This is fine for a tiny smoke-test dataset but will be very slow "
                "for anything larger."
            )

        run_dir = self._run_dir(run_id)
        adapter_dir = os.path.join(run_dir, "adapter")

        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs = {}
            if use_qlora:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
            if use_qlora:
                model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=DEFAULT_LORA_DROPOUT,
                target_modules=target_modules or DEFAULT_TARGET_MODULES,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

            data_files = {"train": run["train_path"]}
            if run.get("val_path"):
                data_files["validation"] = run["val_path"]
            dataset = load_dataset("json", data_files=data_files)

            def format_and_tokenize(example):
                # Concatenate prompt + completion as one sequence — the
                # standard "causal LM fine-tuning on instruction pairs"
                # framing, where the model learns to continue the prompt
                # with the completion. EOS marks where to stop generating.
                text = example["prompt"] + example["completion"] + tokenizer.eos_token
                return tokenizer(text, truncation=True, max_length=512)

            tokenized = dataset.map(format_and_tokenize, remove_columns=["prompt", "completion"])

            training_args = TrainingArguments(
                output_dir=os.path.join(run_dir, "checkpoints"),
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                logging_steps=10,
                save_strategy="no",     # only the final adapter matters here, not intermediate checkpoints
                report_to=[],            # no wandb/tensorboard by default — keep this self-contained
                fp16=cuda_available,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized["train"],
                eval_dataset=tokenized.get("validation"),
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            train_result = trainer.train()

            os.makedirs(adapter_dir, exist_ok=True)
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)

        except Exception as e:
            self._runs[run_id]["status"] = "failed"
            return f"Error during fine-tuning run '{run_id}': {type(e).__name__}: {e}"

        self._runs[run_id].update({
            "status": "trained",
            "adapter_path": adapter_dir,
            "base_model_id": base_model_id,
            "use_qlora": use_qlora,
            "final_loss": getattr(train_result, "training_loss", None),
        })

        loss_str = f"{train_result.training_loss:.4f}" if hasattr(train_result, "training_loss") else "?"
        return (
            f"Fine-tuning complete for run '{run_id}'.\n"
            f"  base model: {base_model_id}\n"
            f"  method: {'QLoRA (4-bit)' if use_qlora else 'LoRA'}\n"
            f"  final training loss: {loss_str}\n"
            f"  adapter saved to: workspace/{FINETUNE_SUBDIR}/{run_id}/adapter/\n"
            f"Next: call merge_and_export('{run_id}') to produce a standalone model "
            f"directory, or load the adapter directly with PEFT's "
            f"PeftModel.from_pretrained(base_model, adapter_path)."
        )

    # ───────────────────────────────────────────────── 3. merge & export

    def merge_and_export(self, run_id: str, export_mode: str = "merged") -> str:
        """
        export_mode="merged"  — fold the LoRA weights into the base
          model's weights and save a single standalone model directory.
          Larger on disk (full model size again), but loadable by
          anything that just expects a normal HuggingFace model — no
          PEFT-awareness needed downstream.
        export_mode="adapter" — leave the adapter as-is (already saved
          by fine_tune) and just confirm/report its location. Smaller on
          disk, but the loading code needs to know to apply it via PEFT.
        """
        run = self._runs.get(run_id)
        if run is None or run.get("status") != "trained":
            return (
                f"Error: run '{run_id}' has not completed training "
                f"(status: {run.get('status') if run else 'not found'}). "
                f"Call fine_tune first."
            )

        if export_mode == "adapter":
            return (
                f"Run '{run_id}' adapter is already saved at "
                f"workspace/{FINETUNE_SUBDIR}/{run_id}/adapter/ — no merge needed. "
                f"Load it with: PeftModel.from_pretrained(AutoModelForCausalLM."
                f"from_pretrained('{run['base_model_id']}'), '<adapter_path>')"
            )

        if export_mode != "merged":
            return f"Error: export_mode must be 'merged' or 'adapter', got '{export_mode}'."

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            return f"Error: merging requires 'pip install transformers peft'. Missing: {e}"

        run_dir = self._run_dir(run_id)
        merged_dir = os.path.join(run_dir, "merged")

        try:
            base_model = AutoModelForCausalLM.from_pretrained(run["base_model_id"])
            peft_model = PeftModel.from_pretrained(base_model, run["adapter_path"])
            merged_model = peft_model.merge_and_unload()

            os.makedirs(merged_dir, exist_ok=True)
            merged_model.save_pretrained(merged_dir)

            tokenizer = AutoTokenizer.from_pretrained(run["adapter_path"])
            tokenizer.save_pretrained(merged_dir)
        except Exception as e:
            return f"Error merging adapter for run '{run_id}': {type(e).__name__}: {e}"

        self._runs[run_id]["merged_path"] = merged_dir
        return (
            f"Merged adapter into a standalone model at "
            f"workspace/{FINETUNE_SUBDIR}/{run_id}/merged/ — "
            f"loadable with plain AutoModelForCausalLM.from_pretrained(), "
            f"no PEFT required downstream."
        )

    # ───────────────────────────────────────────────── inspection

    def list_finetune_runs(self) -> str:
        if not self._runs:
            return "No fine-tuning runs yet. Call prepare_dataset(examples, run_id=...) to start one."
        lines = ["Fine-tuning runs:"]
        for run_id, info in self._runs.items():
            lines.append(f"  {run_id}: {info.get('status')}" + (
                f"  (base: {info['base_model_id']}, loss: {info.get('final_loss', '?')})"
                if info.get("base_model_id") else ""
            ))
        return "\n".join(lines)

    # ───────────────────────────────────────────────── internals

    def _run_dir(self, run_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in run_id)
        return os.path.join(WORKSPACE_DIR, FINETUNE_SUBDIR, safe)


# ─── singleton, matching the rest of the codebase ──────────────────────────────

_tuner: Optional[FineTuner] = None


def get_fine_tuner() -> FineTuner:
    global _tuner
    if _tuner is None:
        _tuner = FineTuner()
    return _tuner
