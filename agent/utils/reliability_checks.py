"""
Reliability checks for training scripts.
"""

import re


def check_training_script_save_pattern(script: str) -> str | None:
    """Check if a training script has proper model saving patterns."""
    if not script:
        return None

    # Check for model.save or torch.save patterns
    save_patterns = [
        r"\.save\(",
        r"torch\.save\(",
        r"model\.save_pretrained\(",
        r"trainer\.save_model\(",
    ]

    has_save = any(re.search(pattern, script) for pattern in save_patterns)

    if not has_save:
        return (
            "[yellow]⚠ Reliability Check:[/yellow] Script doesn't appear to save "
            "the trained model. Consider adding model.save(), torch.save(), "
            "or trainer.save_model() to persist your model."
        )

    return None