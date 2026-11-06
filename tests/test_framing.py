"""The framed msgpack codec carries plain data and numeric arrays, and nothing else.

The gateway decodes every frame a submission sends, so a frame must never be able
to run code or build objects out of attacker-chosen bytes.
"""

from __future__ import annotations

import pickle
import socket
import struct

import msgpack
import numpy as np
import pytest

from capx.protocol.framing import MalformedFrameError, packb, recv_framed, send_framed, unpackb

RAN: list[str] = []


def record_run(message: str) -> None:
    RAN.append(message)


class Marker:
    def __reduce__(self):
        return (record_run, ("pickle ran",))


def test_the_canary_detects_unpickling() -> None:
    pickle.loads(pickle.dumps(Marker()))
    assert RAN == ["pickle ran"]
    RAN.clear()


def _frame(obj: object) -> bytes:
    payload = msgpack.packb(obj, use_bin_type=True)
    return struct.pack("!I", len(payload)) + payload


def _recv(frame: bytes) -> dict:
    a, b = socket.socketpair()
    try:
        a.sendall(frame)
        return recv_framed(b)
    finally:
        a.close()
        b.close()


def test_numeric_arrays_and_scalars_round_trip() -> None:
    message = {
        "rgb": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "depth": np.linspace(0, 1, 6, dtype=np.float32).reshape(2, 3),
        "pose": np.asfortranarray(np.eye(3)),
        "mask": np.array([True, False]),
        "empty": np.zeros((0, 3), dtype=np.int64),
        "scalar": np.float32(1.5),
        "nested": [{"joints": np.zeros(7)}],
    }
    decoded = unpackb(packb(message))
    for key in ("rgb", "depth", "pose", "mask", "empty"):
        np.testing.assert_array_equal(decoded[key], message[key])
        assert decoded[key].dtype == message[key].dtype
    assert decoded["scalar"] == np.float32(1.5)
    np.testing.assert_array_equal(decoded["nested"][0]["joints"], np.zeros(7))


def test_arrays_from_a_msgpack_numpy_peer_decode() -> None:
    msgpack_numpy = pytest.importorskip("msgpack_numpy")
    joints = np.linspace(-1, 1, 7)
    payload = msgpack.packb({"joints": joints}, default=msgpack_numpy.encode, use_bin_type=True)
    np.testing.assert_array_equal(unpackb(payload)["joints"], joints)


def test_a_pickled_object_array_is_refused_without_unpickling() -> None:
    forged = {b"nd": True, b"kind": b"O", b"type": "|O", b"shape": [1], b"data": pickle.dumps(Marker())}
    with pytest.raises(MalformedFrameError):
        _recv(_frame({"method": "step", "payload": forged}))
    assert RAN == []


@pytest.mark.parametrize(
    "array",
    [
        # An object dtype under any other kind would make numpy read pointers from the bytes.
        {b"nd": True, b"kind": b"", b"type": "|O", b"shape": [1], b"data": b"\x00" * 8},
        {b"nd": True, b"kind": b"V", b"type": [["a", "<f8"]], b"shape": [1], b"data": b"\x00" * 8},
        {b"nd": True, b"kind": b"", b"type": "<U4", b"shape": [1], b"data": b"\x00" * 16},
        {b"nd": True, b"kind": b"", b"type": "(2,)<f8", b"shape": [1], b"data": b"\x00" * 16},
        {b"nd": True, b"kind": b"", b"type": "<f8", b"shape": [3], b"data": b"\x00" * 8},
        {b"nd": True, b"kind": b"", b"type": "<f8", b"shape": [-1], b"data": b"\x00" * 8},
        {b"nd": True, b"kind": b"", b"type": "<f8", b"shape": "1", b"data": b"\x00" * 8},
        {b"nd": False, b"type": "|O", b"data": b"\x00" * 8},
        {b"nd": False, b"type": "<f8", b"data": b"\x00" * 3},
    ],
)
def test_anything_but_a_plain_numeric_array_is_refused(array: dict) -> None:
    with pytest.raises(MalformedFrameError):
        _recv(_frame({"payload": array}))


def test_arrays_decode_when_strings_stay_bytes() -> None:
    joints = np.linspace(-1, 1, 7)
    decoded = unpackb(packb({"joints": joints}), raw=True)
    np.testing.assert_array_equal(decoded[b"joints"], joints)


def test_array_rank_is_bounded_before_the_size_check() -> None:
    forged = {b"nd": True, b"kind": b"", b"type": "<f8", b"shape": [2**62] * 1000, b"data": b""}
    with pytest.raises(MalformedFrameError, match="shape"):
        _recv(_frame({"payload": forged}))


def test_a_frame_that_is_not_a_map_is_refused() -> None:
    with pytest.raises(MalformedFrameError):
        _recv(_frame([1, 2, 3]))


def test_garbage_bytes_are_a_malformed_frame() -> None:
    with pytest.raises(MalformedFrameError):
        _recv(struct.pack("!I", 3) + b"\xc1\xc1\xc1")


def test_sending_an_object_array_fails_on_the_sender() -> None:
    a, b = socket.socketpair()
    try:
        with pytest.raises(TypeError):
            send_framed(a, {"x": np.array([object()])})
    finally:
        a.close()
        b.close()


def test_a_deadline_read_restores_the_socket_timeout() -> None:
    import time

    a, b = socket.socketpair()
    try:
        b.settimeout(None)
        a.sendall(_frame({"x": 1}))
        assert recv_framed(b, deadline=time.monotonic() + 5) == {"x": 1}
        assert b.gettimeout() is None
        a.sendall(struct.pack("!I", 3) + b"\xc1\xc1\xc1")
        with pytest.raises(MalformedFrameError):
            recv_framed(b, deadline=time.monotonic() + 5)
        assert b.gettimeout() is None
    finally:
        a.close()
        b.close()
