"""Threaded msgpack-socket server exposing a :class:`Gateway` on an AF_UNIX socket.

One connection is one lane: each accepted connection claims a ``GatewaySession``
for its lifetime and hands it back on disconnect, so a submission runs N episodes
at once by opening N connections. A connection past the lane cap is answered with
the reason rather than dropped, unless more than ``max_connections`` are open at
once, which only a submission flooding the socket does. ``close`` from a submission
retires its lane and ends the connection.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time

from capx.gateway.gateway import Gateway
from capx.protocol.env_protocol import dispatch_to_wire
from capx.protocol.framing import (
    MalformedFrameError,
    bind_unix_socket,
    recv_framed,
    send_framed,
)

# A request is an action or a task id; observations only flow the other way.
MAX_REQUEST_BYTES = 1024 * 1024

# How long a connection past the lane cap may take to send the request it is
# refused on, and a closing connection to take the reply to its close.
REFUSAL_TIMEOUT_S = 5.0

# How long a new connection waits for a free connection slot before it is dropped.
SLOT_WAIT_S = 1.0


class GatewayServer:
    def __init__(
        self,
        gateway: Gateway,
        path: str,
        *,
        max_connections: int = 64,
    ) -> None:
        self._gateway = gateway
        self.path = path
        self._connection_slots = threading.BoundedSemaphore(max_connections)
        self._sock = bind_unix_socket(path)
        self._stop = threading.Event()

    def start_background(self) -> GatewayServer:
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def serve_forever(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                break
            # A handler that just answered close frees its slot a moment after the
            # client sees the reply, so a prompt reconnect waits for it briefly.
            if not self._connection_slots.acquire(timeout=SLOT_WAIT_S):
                conn.close()
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            with conn:
                self._serve(conn)
        finally:
            self._connection_slots.release()

    def _serve(self, conn: socket.socket) -> None:
        try:
            lane = self._gateway.acquire()
        except ValueError as exc:
            # An unexplained EOF would look like a crashed gateway.
            with contextlib.suppress(ConnectionError, EOFError, OSError, MalformedFrameError):
                recv_framed(
                    conn,
                    max_bytes=MAX_REQUEST_BYTES,
                    deadline=time.monotonic() + REFUSAL_TIMEOUT_S,
                )
                send_framed(conn, {"ok": False, "error": None, "detail": str(exc)})
            return
        released = False
        try:
            while not self._stop.is_set():
                try:
                    request = recv_framed(conn, max_bytes=MAX_REQUEST_BYTES)
                except MalformedFrameError as exc:
                    send_framed(conn, {"ok": False, "error": None, "detail": str(exc)})
                    continue
                method = request.get("method", "")
                reply = dispatch_to_wire(lane, method, request.get("payload", {}), jpeg_rgb=True)
                if isinstance(method, str) and method == "close":
                    # Hand the lane back before answering, so a client that reconnects
                    # right after close() finds it free. The connection slot stays held
                    # until the reply is out, which a peer that stops reading cannot delay
                    # for long.
                    self._gateway.release(lane)
                    released = True
                    conn.settimeout(REFUSAL_TIMEOUT_S)
                    send_framed(conn, reply)
                    return
                send_framed(conn, reply)
        except (ConnectionError, EOFError, OSError):
            pass
        finally:
            if not released:
                with contextlib.suppress(Exception):
                    self._gateway.release(lane)

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        with contextlib.suppress(OSError):
            os.unlink(self.path)


__all__ = ["GatewayServer"]
