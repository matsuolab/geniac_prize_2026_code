"""The RPC seam: ``EnvProtocol`` (what the gateway serves), ``Transport`` and ``EnvClient``.

- ``InProcessTransport`` — client and gateway lane in one process (dev).
- ``HttpTransport`` — framed msgpack over TCP (two containers).
- ``UnixTransport`` — the same wire over an AF_UNIX socket, for a submission
  container with no network stack (eval queue).

Every transport runs the same ``dispatch_to_wire``, so dev and official behave
identically and the client can only reset / step / get_observation / finish.
"""

from __future__ import annotations

import enum
import logging
import socket
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from capx.protocol.actions import InvalidActionError, RawAction, decode_action, encode_action
from capx.protocol.observations import FullObs, ProprioObs, decode_obs, encode_obs


class ErrorCode(str, enum.Enum):
    ENV_CRASH = "env_crash"  # sim-side failure: episode voided, trial refunded
    CLIENT_TIMEOUT = "client_timeout"  # call for an episode that is over or unknown
    INVALID_ACTION = "invalid_action"  # action refused; the trial continues
    TRIALS_EXHAUSTED = "trials_exhausted"  # reset refused: no trials left for this task
    JOB_TIMEOUT = "job_timeout"  # whole-job budget reached
    JOB_FINISHED = "job_finished"  # the submission called finish(); the job is over


class GatewayError(RuntimeError):
    """Carries an :class:`ErrorCode` back across the client boundary."""

    def __init__(self, code: ErrorCode, detail: str = "") -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


class RequestError(ValueError):
    """A request the gateway cannot read or serve as asked: a malformed payload, an
    unknown method or task. Its text only echoes the client's own input, so it
    crosses to the client uncoded."""


@dataclass
class ResetResult:
    episode_id: str
    obs: FullObs
    instruction: str


@dataclass
class StepResult:
    """Result of one ``step``. Reward and success never cross this boundary."""

    obs: ProprioObs | FullObs  # proprio tier, or full tier if include_cameras
    terminated: bool
    truncated: bool


@runtime_checkable
class EnvProtocol(Protocol):
    """The complete environment surface. Implemented by a gateway lane."""

    def manifest(self) -> list[dict[str, Any]]:
        """List attemptable tasks (``ManifestEntry`` dicts). No grading info."""
        ...

    def reset(self, task_id: str) -> ResetResult:
        """Begin an episode. The gateway allocates the seed internally and does
        not disclose it; trial accounting is charged here."""
        ...

    def step(self, episode_id: str, action: RawAction) -> StepResult:
        """Advance ``action.repeat`` control cycles."""
        ...

    def get_observation(self, episode_id: str) -> FullObs:
        """Return the full obs tier (cameras + proprio) on demand."""
        ...

    def finish(self) -> None:
        """End the whole job from any lane: the clock stops and the score is final.
        Later reset / step / get_observation calls get ``job_finished``."""
        ...

    def close(self) -> None:
        ...


_logger = logging.getLogger(__name__)


class _TaskRequest(BaseModel):
    task_id: str


class _EpisodeRequest(BaseModel):
    episode_id: str


class _StepRequest(_EpisodeRequest):
    # Checked on the invalid_action path instead: a malformed action is not a
    # malformed request.
    action: Any


_RequestT = TypeVar("_RequestT", bound=BaseModel)


def _read(model: type[_RequestT], payload: Any) -> _RequestT:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise RequestError(str(exc)) from exc


def dispatch_to_wire(
    gateway: EnvProtocol, method: str, payload: Any, *, jpeg_rgb: bool = False
) -> dict[str, Any]:
    """Decode one request, call ``gateway``, and encode the reply as
    ``{"ok", "result" | "error"/"detail"}``.

    The one server-side chokepoint for every transport, so errors are identical in
    dev and official:

    - ``GatewayError`` crosses with its ``ErrorCode`` and detail;
    - ``RequestError`` (the request could not be read, or names a task the gateway
      does not have) only echoes the client's own input, so it crosses uncoded;
    - anything else is logged here and crosses as a fixed generic detail, never the
      exception text.

    ``jpeg_rgb`` shrinks RGB on the socket path; in-process keeps it as numpy.
    """
    try:
        if method == "manifest":
            result: dict[str, Any] = {"tasks": gateway.manifest()}
        elif method == "reset":
            reset = gateway.reset(_read(_TaskRequest, payload).task_id)
            result = {
                "episode_id": reset.episode_id,
                "instruction": reset.instruction,
                "obs": encode_obs(reset.obs, jpeg_rgb=jpeg_rgb),
            }
        elif method == "step":
            request = _read(_StepRequest, payload)
            try:
                if not isinstance(request.action, dict):
                    raise InvalidActionError(
                        f"action must be a map, got {type(request.action).__name__}"
                    )
                action = decode_action(request.action)
            except InvalidActionError as exc:
                raise GatewayError(ErrorCode.INVALID_ACTION, str(exc)) from exc
            stepped = gateway.step(request.episode_id, action)
            result = {
                "obs": encode_obs(stepped.obs, jpeg_rgb=jpeg_rgb),
                "terminated": stepped.terminated,
                "truncated": stepped.truncated,
            }
        elif method == "get_observation":
            obs = gateway.get_observation(_read(_EpisodeRequest, payload).episode_id)
            result = {"obs": encode_obs(obs, jpeg_rgb=jpeg_rgb)}
        elif method == "finish":
            gateway.finish()
            result = {}
        elif method == "close":
            gateway.close()
            result = {}
        else:
            raise RequestError(f"unknown method: {method!r}")
    except GatewayError as exc:
        return {"ok": False, "error": exc.code.value, "detail": exc.detail}
    except RequestError as exc:
        return {"ok": False, "error": None, "detail": f"{type(exc).__name__}: {exc}"}
    except Exception:
        _logger.exception("unhandled gateway error in %s", method)
        return {"ok": False, "error": None, "detail": "internal gateway error (see gateway log)"}
    return {"ok": True, "result": result}


def result_from_wire(resp: dict[str, Any]) -> dict[str, Any]:
    """Client-side inverse of :func:`dispatch_to_wire`: unwrap or re-raise."""
    if not resp.get("ok", False):
        code = resp.get("error")
        detail = resp.get("detail", "")
        if code in ErrorCode._value2member_map_:
            raise GatewayError(ErrorCode(code), detail)
        raise RuntimeError(detail)
    return resp["result"]


class Transport(Protocol):
    """Carries one RPC call to an ``EnvProtocol`` implementation."""

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        ...


@runtime_checkable
class ConnectionTransport(Transport, Protocol):
    """A transport over a connection that :meth:`EnvClient.close` should end."""

    @property
    def connected(self) -> bool:
        """Whether a call would reach a lane without opening a new connection."""
        ...

    def close(self) -> None:
        ...


class InProcessTransport:
    """Dev transport: call a gateway lane directly, without serialization."""

    def __init__(self, gateway: EnvProtocol) -> None:
        self._gateway = gateway

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        return result_from_wire(dispatch_to_wire(self._gateway, method, payload))


class _FramedTransport:
    """Framed msgpack over a stream socket. ``framing`` is imported lazily so
    the in-process path never needs msgpack; subclasses supply how to dial."""

    def __init__(self) -> None:
        self._sock: socket.socket | None = None

    def _open(self) -> socket.socket:
        raise NotImplementedError

    def _connection(self) -> socket.socket:
        if self._sock is None:
            self._sock = self._open()
        return self._sock

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        from capx.protocol.framing import recv_framed, send_framed

        sock = self._connection()
        try:
            send_framed(sock, {"method": method, "payload": payload})
            reply = recv_framed(sock)
        finally:
            if method == "close":
                self.close()  # the gateway ends the connection after close
        return result_from_wire(reply)

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class HttpTransport(_FramedTransport):
    """Official transport over TCP — two containers sharing a host network."""

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self._host = host
        self._port = port

    def _open(self) -> socket.socket:
        from capx.protocol.framing import connect_stream_socket

        return connect_stream_socket(host=self._host, port=self._port)


class UnixTransport(_FramedTransport):
    """Official transport over an AF_UNIX socket bind-mounted into a submission
    container that has no network stack (``--network=none``)."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def _open(self) -> socket.socket:
        from capx.protocol.framing import connect_stream_socket

        return connect_stream_socket(unix_path=self._path)


class EnvClient:
    """Submission-side :class:`EnvProtocol` over a transport."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def manifest(self) -> list[dict[str, Any]]:
        return self._transport.call("manifest", {})["tasks"]

    def reset(self, task_id: str) -> ResetResult:
        resp = self._transport.call("reset", {"task_id": task_id})
        obs = decode_obs(resp["obs"])
        assert isinstance(obs, FullObs)
        return ResetResult(episode_id=resp["episode_id"], obs=obs, instruction=resp["instruction"])

    def step(self, episode_id: str, action: RawAction) -> StepResult:
        resp = self._transport.call(
            "step", {"episode_id": episode_id, "action": encode_action(action)}
        )
        return StepResult(
            obs=decode_obs(resp["obs"]),
            terminated=resp["terminated"],
            truncated=resp["truncated"],
        )

    def get_observation(self, episode_id: str) -> FullObs:
        resp = self._transport.call("get_observation", {"episode_id": episode_id})
        obs = decode_obs(resp["obs"])
        assert isinstance(obs, FullObs)
        return obs

    def finish(self) -> None:
        self._transport.call("finish", {})

    def close(self) -> None:
        """Retire this lane. Over a connection, also disconnect, so the gateway can hand
        the lane out again; nothing is sent when it is not connected, and a later call
        opens a new lane."""
        transport = self._transport
        if not isinstance(transport, ConnectionTransport):
            transport.call("close", {})
            return
        try:
            if transport.connected:
                transport.call("close", {})
        finally:
            transport.close()


__all__ = [
    "ErrorCode",
    "GatewayError",
    "RequestError",
    "ResetResult",
    "StepResult",
    "EnvProtocol",
    "dispatch_to_wire",
    "result_from_wire",
    "Transport",
    "ConnectionTransport",
    "InProcessTransport",
    "HttpTransport",
    "UnixTransport",
    "EnvClient",
]
