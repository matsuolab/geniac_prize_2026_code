"""The same EnvClient over HttpTransport behaves like in-process.

Runs a GatewayServer in a background thread on an ephemeral port and drives it through
framed msgpack. Confirms reset/step/get_obs work over the socket and that a gateway
error crosses the wire with its original ErrorCode.
"""

from __future__ import annotations

import numpy as np
import pytest

from capx.gateway import EnforcementMode, Gateway, StepOutcome
from capx.gateway.server import GatewayServer
from capx.protocol import (
    EnvClient,
    ErrorCode,
    FrankaRawAction,
    FullObs,
    GatewayError,
    HttpTransport,
    InProcessTransport,
    ProprioObs,
)
from capx.protocol.taskpack import SeedPolicy, TaskSpec

CAMERA = "robot0_robotview"
VALID_JOINTS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])  # within Panda limits


class FakeEnv:
    embodiment = "franka"

    def __init__(self) -> None:
        self.sim_time = 0.0
        self._joints = np.zeros(7)
        self._gripper = 1.0

    def reset(self, *, seed: int) -> None:
        self.sim_time = 0.0

    def apply_raw_action(self, action) -> StepOutcome:
        self.sim_time += 0.05
        self._joints = np.asarray(action.joints)
        self._gripper = float(action.gripper)
        return StepOutcome()

    def get_proprio(self):
        return {
            "robot_joint_pos": np.concatenate([self._joints, [self._gripper]]).astype(np.float32),
            "robot_cartesian_pos": np.zeros(8, dtype=np.float32),
        }

    def get_observation(self):
        obs = self.get_proprio()
        obs["cube_poses"] = {"primary": np.zeros(7, dtype=np.float32)}
        obs[CAMERA] = {
            "images": {"rgb": np.full((4, 4, 3), 7, np.uint8), "depth": np.zeros((4, 4), np.float32)},
            "intrinsics": np.eye(3, dtype=np.float32),
            "pose": np.zeros(7, dtype=np.float32),
        }
        return obs

    def compute_reward(self) -> float:
        return 0.0

    def close(self) -> None:
        pass


class FakeFactory:
    supports_parallel = False

    def build(self, spec) -> FakeEnv:
        return FakeEnv()


def make_tasks(n_trials: int = 2) -> dict[str, TaskSpec]:
    return {
        "t": TaskSpec(
            task_id="t",
            embodiment="franka",
            simulator="robosuite",
            instruction="go",
            camera_names=[CAMERA],
            env_target="fake",
            n_trials=n_trials,
            trial_steplimit=100,
            seed_policy=SeedPolicy(mode="fixed", seeds=[1, 2]),
        )
    }


def serve(mode=EnforcementMode.DEV, n_trials=2) -> GatewayServer:
    gateway = Gateway(make_tasks(n_trials), FakeFactory(), mode)
    return GatewayServer(gateway).start_background()


def test_http_reset_step_get_obs() -> None:
    server = serve()
    client = EnvClient(HttpTransport(server.host, server.port))
    try:
        reset = client.reset("t")
        assert isinstance(reset.obs, FullObs)
        assert reset.obs.cameras[CAMERA].rgb.shape == (4, 4, 3)  # JPEG round-tripped

        result = client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=0.0))
        assert isinstance(result.obs, ProprioObs)
        np.testing.assert_allclose(result.obs.joint_pos[:7], VALID_JOINTS, atol=1e-4)

        full = client.get_observation(reset.episode_id)
        assert CAMERA in full.cameras
    finally:
        server.close()


def test_http_matches_inprocess_for_the_same_action() -> None:
    server = serve()
    http = EnvClient(HttpTransport(server.host, server.port))
    inproc = EnvClient(
        InProcessTransport(Gateway(make_tasks(), FakeFactory(), EnforcementMode.DEV).acquire())
    )
    try:
        action = FrankaRawAction(joints=VALID_JOINTS, gripper=0.5)
        h = http.step(http.reset("t").episode_id, action)
        i = inproc.step(inproc.reset("t").episode_id, action)
        np.testing.assert_allclose(h.obs.joint_pos, i.obs.joint_pos, atol=1e-4)
    finally:
        server.close()


def test_http_propagates_gateway_error() -> None:
    server = serve(EnforcementMode.OFFICIAL, n_trials=1)
    client = EnvClient(HttpTransport(server.host, server.port))
    try:
        client.reset("t")
        with pytest.raises(GatewayError) as exc:
            client.reset("t")  # trial 2 of a 1-trial task
        assert exc.value.code is ErrorCode.TRIALS_EXHAUSTED
    finally:
        server.close()


def test_malformed_action_over_the_wire_is_invalid_action() -> None:
    server = serve()
    transport = HttpTransport(server.host, server.port)
    try:
        ep = EnvClient(transport).reset("t").episode_id
        with pytest.raises(GatewayError) as exc:
            transport.call(
                "step", {"episode_id": ep, "action": {"embodiment": "franka", "gripper": 0.5}}
            )
        assert exc.value.code is ErrorCode.INVALID_ACTION
    finally:
        transport.close()
        server.close()


def test_noncoded_errors_match_across_transports() -> None:
    server = serve()
    http = EnvClient(HttpTransport(server.host, server.port))
    inproc = EnvClient(
        InProcessTransport(Gateway(make_tasks(), FakeFactory(), EnforcementMode.DEV).acquire())
    )
    try:
        with pytest.raises(RuntimeError) as over_wire:
            http.reset("nope")
        with pytest.raises(RuntimeError) as in_process:
            inproc.reset("nope")
        assert str(over_wire.value) == str(in_process.value)
        assert type(over_wire.value) is type(in_process.value)
    finally:
        server.close()


def test_frame_size_is_capped_both_directions() -> None:
    import socket
    import struct

    from capx.protocol.framing import MAX_FRAME_BYTES, recv_framed, send_framed

    a, b = socket.socketpair()
    try:
        with pytest.raises(ValueError, match="MAX_FRAME_BYTES"):
            send_framed(a, {"blob": b"x" * (MAX_FRAME_BYTES + 1)})
        # A hostile peer declaring a giant frame is rejected before any buffering.
        a.sendall(struct.pack("!I", MAX_FRAME_BYTES + 1))
        with pytest.raises(ConnectionError, match="bytes allowed"):
            recv_framed(b)
    finally:
        a.close()
        b.close()
