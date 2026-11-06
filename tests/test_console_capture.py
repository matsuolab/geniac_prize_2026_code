"""Console capture has to survive trials running at the same time.

The multiturn loop feeds the executed code's stdout and stderr back to the model as
the result of what it wrote. ``contextlib.redirect_stdout`` swaps a process-global,
so with concurrent lanes one trial would be shown another's output — and, worse,
would restore whatever stream it happened to find on the way out.

Needs the full env stack (gymnasium), so it runs through
``scripts/evalq/test-in-image.sbatch`` rather than from a bare checkout.
"""

from __future__ import annotations

import sys
import threading

import pytest

# Skips rather than fails on a bare checkout: capx.envs.tasks.base pulls gymnasium,
# which only exists inside a built image. Collecting it unconditionally would break
# the local suite for everyone who is not running through the image.
pytest.importorskip("gymnasium")

from capx.envs.tasks.base import _console_capture  # noqa: E402


def test_each_thread_sees_only_its_own_output() -> None:
    out, _ = _console_capture()
    captured: dict[str, str] = {}
    ready = threading.Barrier(3, timeout=10.0)

    def work(name: str) -> None:
        with out.capturing() as buffer:
            ready.wait()  # every thread is inside a capture at the same time
            print(f"from-{name}")
            ready.wait()
            captured[name] = buffer.getvalue()

    threads = [threading.Thread(target=work, args=(name,)) for name in ("a", "b", "c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert captured == {"a": "from-a\n", "b": "from-b\n", "c": "from-c\n"}


def test_a_thread_outside_a_capture_still_reaches_the_console() -> None:
    out, _ = _console_capture()
    with out.capturing() as buffer:
        print("mine")
    print("not mine")
    assert buffer.getvalue() == "mine\n"


def test_nested_captures_restore_the_outer_one() -> None:
    out, _ = _console_capture()
    with out.capturing() as outer:
        print("outer-before")
        with out.capturing() as inner:
            print("inner")
        print("outer-after")

    assert inner.getvalue() == "inner\n"
    assert outer.getvalue() == "outer-before\nouter-after\n"


def test_the_proxy_answers_for_the_real_stream() -> None:
    """It is installed process-wide and stays there, so anything asking sys.stdout
    a question -- a descriptor for a subprocess, an encoding -- must get the real one."""
    console = sys.__stdout__
    out, _ = _console_capture()
    assert out.fileno() == console.fileno()
    assert out.encoding == console.encoding
    assert out.buffer is console.buffer


def test_installing_twice_does_not_stack_proxies() -> None:
    """A proxy wrapping a proxy would double every line on the console."""
    first, _ = _console_capture()
    second, _ = _console_capture()
    assert first is second
    assert sys.stdout is first
