"""
Convenience entry point: `python main.py` opens the interactive REPL.

There used to be a second, hand-rolled REPL here — a `while True` around
`input("you> ")` with its own command table, its own banner, and no approval
prompts. It had drifted from agent/cli.py's REPL: commands existed in one and
not the other, and a fix to either left the other stale. There is now one
implementation, in agent/cli.py, and this file is a thin shim to it.

The real front end lives in the package rather than at the repo root because
the `swarn` console script has to be importable from an installed copy: a
root-level module resolves fine from a source checkout and raises
ModuleNotFoundError once installed, since it is not part of the `agent`
package. Prefer either of these:

    swarn                    # installed (pip install -e .)
    python -m agent.cli      # source checkout, nothing installed

Both accept the same arguments as this file, which simply forwards argv:

    python main.py                      # interactive REPL
    python main.py "build me a model"   # headless one-shot
    python main.py team "..."           # headless multi-agent pipeline
    python main.py --help               # every command
"""

from agent.cli import main

if __name__ == "__main__":
    main()
