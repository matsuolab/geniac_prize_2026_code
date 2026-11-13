from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass

import tyro

from capx.utils.unix_http import ensure_private_dir, is_listening, resolve_socket


@dataclass
class Args:
    """Arguments for launching the vLLM OpenAI-compatible server."""

    model: str
    socket: str = "model.sock"
    """Unix socket to serve on: a path, or a name inside $CAPX_SOCKET_DIR."""
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    download_dir: str | None = None
    log_file: str | None = None


def _build_command(args: Args) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--uds",
        str(resolve_socket(args.socket)),
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.download_dir:
        cmd.extend(["--download-dir", args.download_dir])
    return cmd


def main(args: Args) -> None:
    try:
        import vllm  # noqa: F401  # pylint: disable=unused-import
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "vllm is required. Install with `pip install vllm` or add to extras."
        ) from exc

    socket_path = resolve_socket(args.socket)
    ensure_private_dir(socket_path.parent)
    # vLLM binds the path itself and fails on a file a dead server left behind.
    if socket_path.is_socket():
        if is_listening(socket_path):
            raise SystemExit(f"another server is already listening on {socket_path}")
        socket_path.unlink()
    cmd = _build_command(args)
    env = os.environ.copy()

    log_handle: int | None = None
    stdout = stderr = None
    if args.log_file:
        log_handle = os.open(args.log_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        stdout = stderr = log_handle

    previous_umask = os.umask(0o077)
    try:
        process = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
    finally:
        os.umask(previous_umask)

    def _forward_signal(signum: int, _frame: object | None) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    try:
        rc = process.wait()
    finally:
        if log_handle is not None:
            os.close(log_handle)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main(tyro.cli(Args))
