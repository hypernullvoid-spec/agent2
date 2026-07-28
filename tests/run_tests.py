"""
Zero-dependency test runner — `python tests/run_tests.py` runs every
test_* function in this directory without needing pytest installed.
"""

import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODULES = [
    "test_llm_layer",
    "test_execution",
    "test_journal",
    "test_search_agent",
    "test_static_check",
    "test_knowledge",
    "test_doom_loop",
    "test_e2e_search",
    "test_parallel_resume",
]


def main() -> int:
    sys.path.insert(0, os.path.dirname(__file__))
    passed = failed = 0
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"  PASS  {mod_name}.{name}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"  FAIL  {mod_name}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
