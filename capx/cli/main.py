"""``capx`` console entry point: ``capx eval`` / ``capx evalq`` are
``python -m capx.cli.eval`` / ``python -m capx.cli.evalq`` with the same flags."""

from __future__ import annotations

import sys

import tyro

from capx.cli.eval import EvalArgs
from capx.cli.eval import main as eval_main


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "eval":
        eval_main(tyro.cli(EvalArgs, args=argv[1:], prog="capx eval"))
        return
    if argv and argv[0] == "evalq":
        # Imported lazily: `capx eval` should not pay for the queue's imports.
        from capx.cli.evalq import main as evalq_main

        evalq_main(argv[1:])
        return
    raise SystemExit(
        "usage: capx eval [options] | capx evalq <subcommand> [options]  "
        "(see 'capx eval --help' / 'capx evalq --help')"
    )


if __name__ == "__main__":
    main()
