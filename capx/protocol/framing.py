"""Blocking length-framed msgpack (4-byte big-endian length + msgpack, string keys),
and the TCP / AF_UNIX stream sockets both framed links run on (submission<->gateway,
gateway<->env backend).

Numpy arrays travel in msgpack-numpy's layout, so a peer using msgpack-numpy can
still read and write them. Decoding is our own: msgpack-numpy unpickles object
arrays, and the gateway decodes frames from untrusted submissions.
"""

from __future__ import annotations

import contextlib
import math
import os
import socket
import stat
import struct
import time
from typing import Any

import msgpack
import numpy as np

# sockaddr_un.sun_path is 108 bytes including the NUL on Linux. Bind fails with a
# bare ENAMETOOLONG otherwise, which is a miserable thing to debug from a job log.
MAX_UNIX_PATH_BYTES = 107

# Backstop against a hostile/buggy peer declaring a huge frame and exhausting
# server memory: the largest legitimate frame is a full-tier obs (a few raw
# 512x512 RGB-D views on the backend link), well under this.
MAX_FRAME_BYTES = 64 * 1024 * 1024

_ARRAY_KINDS = frozenset("biuf")
_MAX_ARRAY_DIMS = 32


class MalformedFrameError(ValueError):
    """A complete frame arrived but is not a message this protocol can carry.

    The frame was read in full, so the stream is still in sync and the connection
    can answer the next request.
    """


def _encode_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind not in _ARRAY_KINDS:
            raise TypeError(f"cannot send a numpy array of dtype {obj.dtype}")
        return {
            b"nd": True,
            b"type": obj.dtype.str,
            b"kind": b"",
            b"shape": list(obj.shape),
            b"data": obj.tobytes(),
        }
    if isinstance(obj, np.bool_ | np.number):
        if obj.dtype.kind not in _ARRAY_KINDS:
            raise TypeError(f"cannot send a numpy scalar of dtype {obj.dtype}")
        return {b"nd": False, b"type": obj.dtype.str, b"data": obj.tobytes()}
    raise TypeError(f"cannot send an object of type {type(obj).__name__}")


def _numeric_dtype(descr: Any) -> np.dtype:
    if isinstance(descr, bytes):
        try:
            descr = descr.decode("ascii")
        except UnicodeDecodeError as exc:
            raise MalformedFrameError("array dtype is not ASCII") from exc
    if not isinstance(descr, str):
        raise MalformedFrameError(f"array dtype must be a string, got {type(descr).__name__}")
    try:
        dtype = np.dtype(descr)
    except TypeError as exc:
        raise MalformedFrameError(f"unknown array dtype {descr!r}") from exc
    if dtype.kind not in _ARRAY_KINDS or dtype.hasobject or dtype.fields is not None:
        raise MalformedFrameError(f"array dtype {descr!r} is not a plain numeric type")
    return dtype


def _decode_numpy(obj: dict[Any, Any]) -> Any:
    is_array = obj.get(b"nd")
    if is_array is None:
        return obj
    dtype = _numeric_dtype(obj.get(b"type"))
    data = obj.get(b"data")
    if not isinstance(data, bytes):
        raise MalformedFrameError("array data must be bytes")
    if is_array is False:
        if len(data) != dtype.itemsize:
            raise MalformedFrameError("scalar data does not match its dtype")
        return np.frombuffer(data, dtype=dtype)[0]
    if is_array is not True:
        raise MalformedFrameError("array marker must be a bool")
    if obj.get(b"kind", b"") != b"":
        raise MalformedFrameError("only plain numeric arrays are accepted")
    shape = obj.get(b"shape")
    if (
        not isinstance(shape, list | tuple)
        or len(shape) > _MAX_ARRAY_DIMS
        or not all(type(dim) is int and dim >= 0 for dim in shape)
    ):
        raise MalformedFrameError("array shape must be a short list of non-negative ints")
    if math.prod(shape) * dtype.itemsize != len(data):
        raise MalformedFrameError("array data does not match its shape and dtype")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, use_bin_type=True, default=_encode_numpy)


def unpackb(payload: bytes, *, raw: bool = False) -> Any:
    """Decode one message. Anything that is not plain data (strings, numbers, bytes,
    lists, maps, numeric arrays) raises :class:`MalformedFrameError`."""
    try:
        return msgpack.unpackb(payload, raw=raw, object_hook=_decode_numpy)
    except MalformedFrameError:
        raise
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise MalformedFrameError(f"undecodable frame: {type(exc).__name__}: {exc}") from exc


def send_framed(sock: socket.socket, obj: dict[str, Any]) -> None:
    payload = packb(obj)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"frame of {len(payload)} bytes exceeds MAX_FRAME_BYTES")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_framed(
    sock: socket.socket, *, max_bytes: int = MAX_FRAME_BYTES, deadline: float | None = None
) -> dict[str, Any]:
    """Read one frame. ``deadline`` (``time.monotonic()``) bounds the whole read, which a
    socket timeout alone does not: that only bounds each ``recv``."""
    timeout = sock.gettimeout()
    try:
        (length,) = struct.unpack("!I", _recvall(sock, 4, deadline))
        if length > max_bytes:
            raise ConnectionError(
                f"peer declared a {length}-byte frame (> {max_bytes} bytes allowed)"
            )
        payload = _recvall(sock, length, deadline)
    finally:
        sock.settimeout(timeout)
    message = unpackb(payload)
    if not isinstance(message, dict):
        raise MalformedFrameError(f"a frame must hold a map, got {type(message).__name__}")
    return message


def _recvall(sock: socket.socket, n: int, deadline: float | None = None) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("frame not received before the deadline")
            sock.settimeout(left)
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def check_unix_path(path: str) -> str:
    """Reject a Unix socket path the kernel would truncate, with a usable message."""
    if len(os.fsencode(path)) > MAX_UNIX_PATH_BYTES:
        raise ValueError(
            f"unix socket path is {len(os.fsencode(path))} bytes, over the "
            f"{MAX_UNIX_PATH_BYTES}-byte kernel limit: {path!r}. Use a shorter run "
            "directory (e.g. node-local scratch) for the sockets."
        )
    return path


def bind_stream_socket(
    *, host: str = "127.0.0.1", port: int = 0, unix_path: str | None = None
) -> tuple[socket.socket, str, int]:
    """Bind + listen on a TCP or AF_UNIX stream socket.

    Returns the socket and a printable address; for a Unix socket the path takes
    the host slot and the port is 0.
    """
    if unix_path is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen()
        bound_host, bound_port = sock.getsockname()
        return sock, bound_host, bound_port

    check_unix_path(unix_path)
    parent = os.path.dirname(unix_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # A socket file left behind by a crashed run makes bind() fail with EADDRINUSE.
    # Only ever unlink an actual socket, so a mistyped path can't delete real data.
    with contextlib.suppress(FileNotFoundError):
        if stat.S_ISSOCK(os.stat(unix_path).st_mode):
            os.unlink(unix_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(unix_path)
    os.chmod(unix_path, 0o600)
    sock.listen()
    return sock, unix_path, 0


def connect_stream_socket(
    *, host: str = "", port: int = 0, unix_path: str | None = None
) -> socket.socket:
    """Dial the TCP or AF_UNIX counterpart of :func:`bind_stream_socket`."""
    if unix_path is None:
        return socket.create_connection((host, port))
    check_unix_path(unix_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(unix_path)
    return sock


__all__ = [
    "MAX_FRAME_BYTES",
    "MAX_UNIX_PATH_BYTES",
    "MalformedFrameError",
    "packb",
    "unpackb",
    "send_framed",
    "recv_framed",
    "check_unix_path",
    "bind_stream_socket",
    "connect_stream_socket",
]
