"""EnvClient -> InProcessTransport -> gateway lane -> fake env, over the real codec.

Verifies the privacy boundary: privileged ground truth never crosses back to the client.
"""

from __future__ import annotations

import numpy as np
import pydantic
import pytest

from capx.gateway import EnforcementMode, Gateway, GatewaySession, StepOutcome
from capx.protocol import (
    EnvClient,
    EnvProtocol,
    ErrorCode,
    FrankaRawAction,
    FullObs,
    GatewayError,
    InvalidActionError,
    ProprioObs,
    InProcessTransport,
    R1ProRawAction,
    TaskSpec,
)
from capx.protocol.env_protocol import dispatch_to_wire
from capx.protocol.taskpack import SeedPolicy

CAMERA = "robot0_robotview"
H, W = 8, 8
DT = 0.05
VALID_JOINTS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])


class FakeFrankaEnv:
    """A GPU-free stand-in for a Robosuite low-level env, matching ``LowLevelEnv``.

    Its ``get_observation`` deliberately injects privileged ``cube_poses`` and GT
    ``segmentation`` so the test can confirm ``ObsFilter`` strips them.
    """

    embodiment = "franka"

    def __init__(self) -> None:
        self.sim_time = 0.0
        self._joints = np.zeros(7, dtype=np.float64)
        self._gripper = 1.0
        self._cycles = 0
        self.closed = False

    def reset(self, *, seed: int) -> None:
        self._joints = VALID_JOINTS.copy()
        self._gripper = 1.0
        self._cycles = 0
        self.sim_time = 0.0

    def apply_raw_action(self, action: FrankaRawAction) -> StepOutcome:
        self._joints = np.asarray(action.joints, dtype=np.float64)
        self._gripper = float(action.gripper)
        self._cycles += 1
        self.sim_time += DT
        return StepOutcome(terminated=self._cycles >= 3, truncated=False)

    def get_proprio(self) -> dict[str, object]:
        return {
            "robot_joint_pos": np.concatenate([self._joints, [self._gripper]]).astype(np.float32),
            "robot_cartesian_pos": np.array(
                [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, self._gripper], dtype=np.float32
            ),
        }

    def get_observation(self) -> dict[str, object]:
        obs = self.get_proprio()
        obs["cube_poses"] = {  # privileged: object ground truth, must be stripped
            "primary": np.zeros(7, dtype=np.float32),
            "secondary": np.ones(7, dtype=np.float32),
        }
        obs[CAMERA] = {
            "images": {
                "rgb": (np.arange(H * W * 3).reshape(H, W, 3) % 256).astype(np.uint8),
                "depth": np.full((H, W), 0.5, dtype=np.float32),
                "segmentation": np.ones((H, W), dtype=np.int32),  # privileged GT mask
            },
            "intrinsics": np.eye(3, dtype=np.float32),
            "pose": np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "pose_mat": np.eye(4, dtype=np.float32),
        }
        return obs

    def compute_reward(self) -> float:
        return 0.0

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    supports_parallel = False

    def __init__(self) -> None:
        self.env = FakeFrankaEnv()

    def build(self, spec: TaskSpec) -> FakeFrankaEnv:
        return self.env


def build_client() -> tuple[EnvClient, FakeFrankaEnv, GatewaySession]:
    factory = FakeFactory()
    task = TaskSpec(
        task_id="stack_cubes",
        embodiment="franka",
        simulator="robosuite",
        instruction="place the red cube on the green cube",
        camera_names=[CAMERA],
        env_target="fake",
        n_trials=1,
        trial_steplimit=100,
    )
    lane = Gateway({"stack_cubes": task}, factory, EnforcementMode.DEV).acquire()
    return EnvClient(InProcessTransport(lane)), factory.env, lane


def action(gripper: float = 1.0, **kw) -> FrankaRawAction:
    return FrankaRawAction(joints=VALID_JOINTS, gripper=gripper, **kw)


def test_a_lane_satisfies_the_protocol() -> None:
    _, _, lane = build_client()
    assert isinstance(lane, EnvProtocol)


def test_manifest_carries_no_grading() -> None:
    client, _, _ = build_client()
    (entry,) = client.manifest()
    assert entry["task_id"] == "stack_cubes"
    assert set(entry) == {
        "task_id",
        "embodiment",
        "instruction",
        "camera_names",
        "action_spec_ref",
        "n_trials",
        "trial_steplimit",
    }


def test_reset_returns_full_obs_and_hides_privileged_state() -> None:
    client, _, lane = build_client()
    result = client.reset("stack_cubes")

    assert isinstance(result.obs, FullObs)
    assert result.instruction == "place the red cube on the green cube"
    assert set(result.obs.cameras) == {CAMERA}
    cam = result.obs.cameras[CAMERA]
    assert cam.rgb.shape == (H, W, 3)
    assert not hasattr(cam, "segmentation")  # GT masks never modeled on the wire

    # Object ground truth exists in the raw env obs but never leaves the gateway.
    assert "cube_poses" in FakeFrankaEnv().get_observation()
    wire = InProcessTransport(lane).call("reset", {"task_id": "stack_cubes"})
    assert set(wire) == {"episode_id", "instruction", "obs"}
    assert "cube_poses" not in _flatten_keys(wire)
    assert "segmentation" not in _flatten_keys(wire)


def test_step_returns_proprio_tier_reflecting_the_action() -> None:
    client, _, _ = build_client()
    ep = client.reset("stack_cubes").episode_id

    result = client.step(ep, action(gripper=0.0))
    assert isinstance(result.obs, ProprioObs)
    np.testing.assert_allclose(result.obs.joint_pos[:7], VALID_JOINTS, atol=1e-5)
    assert result.obs.joint_pos[-1] == pytest.approx(0.0)  # gripper closed


def test_include_cameras_upgrades_step_to_full_tier() -> None:
    client, _, _ = build_client()
    ep = client.reset("stack_cubes").episode_id
    result = client.step(ep, action(include_cameras=True))
    assert isinstance(result.obs, FullObs)
    assert set(result.obs.cameras) == {CAMERA}


def test_repeat_holds_the_command_for_n_cycles() -> None:
    client, _, _ = build_client()
    ep = client.reset("stack_cubes").episode_id
    result = client.step(ep, action(repeat=3))
    # 3 cycles advanced sim_time and tripped the fake env's terminal condition.
    assert result.obs.sim_time == pytest.approx(3 * DT)
    assert result.terminated is True


def test_invalid_action_is_rejected_client_side() -> None:
    client, _, _ = build_client()
    ep = client.reset("stack_cubes").episode_id
    with pytest.raises(InvalidActionError):
        client.step(ep, FrankaRawAction(joints=np.zeros(3), gripper=1.0))


def test_invalid_action_is_rejected_gateway_side() -> None:
    _, _, lane = build_client()
    ep = lane.reset("stack_cubes").episode_id
    bad = {"embodiment": "franka", "joints": np.zeros(3), "gripper": 0.5}
    with pytest.raises(GatewayError) as exc:
        InProcessTransport(lane).call("step", {"episode_id": ep, "action": bad})
    assert exc.value.code is ErrorCode.INVALID_ACTION


def test_close_propagates_to_the_env() -> None:
    client, env, _ = build_client()
    client.reset("stack_cubes")
    client.close()
    assert env.closed is True


def test_missing_action_payload_maps_to_invalid_action() -> None:
    with pytest.raises(InvalidActionError):
        FrankaRawAction(gripper=1.0).validate()
    with pytest.raises(InvalidActionError):
        R1ProRawAction().validate()


@pytest.mark.parametrize(
    "repeat",
    [
        float("inf"),
        float("nan"),
        2.5,
        "3",
        True,
        None,
        np.nextafter(np.longdouble(2), np.longdouble(3)),
    ],
)
def test_a_repeat_that_is_not_an_integer_is_invalid_action(repeat: object) -> None:
    from capx.protocol.actions import decode_action

    msg = {"embodiment": "franka", "joints": np.zeros(7), "gripper": 1.0, "repeat": repeat}
    with pytest.raises(InvalidActionError, match="repeat"):
        decode_action(msg)


@pytest.mark.parametrize("repeat", [2.5, True, float("nan")])
def test_a_repeat_that_is_not_an_integer_fails_client_side(repeat: object) -> None:
    from capx.protocol.actions import encode_action

    with pytest.raises(InvalidActionError, match="repeat"):
        encode_action(FrankaRawAction(joints=np.zeros(7), gripper=1.0, repeat=repeat))


@pytest.mark.parametrize("repeat", [3, 3.0, np.int64(3), np.uint8(3), np.float32(3.0)])
def test_integral_repeats_are_accepted_through_the_codec(repeat: object) -> None:
    from capx.protocol.actions import decode_action
    from capx.protocol.framing import packb, unpackb

    msg = {"embodiment": "franka", "joints": np.zeros(7), "gripper": 1.0, "repeat": repeat}
    assert decode_action(unpackb(packb(msg))).repeat == 3


def test_r1pro_values_must_survive_the_float32_backend_link() -> None:
    from capx.protocol.actions import decode_action

    with pytest.raises(InvalidActionError, match="float32"):
        decode_action({"embodiment": "r1pro", "action": np.array([0.0, 1e100])})
    assert decode_action({"embodiment": "r1pro", "action": np.array([0.0, 1e30])}).action[1] == 1e30


def test_seed_policy_is_validated_at_construction() -> None:
    assert SeedPolicy().mode == "random"
    assert SeedPolicy(mode="fixed", seeds=[1, 2]).seeds == [1, 2]
    with pytest.raises(pydantic.ValidationError):
        SeedPolicy(mode="fixed")
    with pytest.raises(pydantic.ValidationError):
        SeedPolicy(mode="fixed", seeds=[])
    with pytest.raises(pydantic.ValidationError):
        SeedPolicy(mode="sometimes")


def _flatten_keys(obj: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys |= _flatten_keys(v)
    return keys



# --- what an uncoded error may say ------------------------------------------


class _BrokenLane:
    """A lane whose own code fails, with internals in the exception text."""

    def manifest(self) -> list[dict[str, object]]:
        raise KeyError("/opt/gateway/internal_state.py")


def test_an_internal_error_never_echoes_its_text() -> None:
    reply = dispatch_to_wire(_BrokenLane(), "manifest", {})  # type: ignore[arg-type]
    assert reply == {
        "ok": False,
        "error": None,
        "detail": "internal gateway error (see gateway log)",
    }


@pytest.mark.parametrize(
    ("method", "payload", "echoed"),
    [
        ("reset", {}, "task_id"),
        ("reset", {"task_id": "nope"}, "nope"),
        ("step", {"episode_id": "e"}, "action"),
        ("get_observation", [], "valid dictionary"),
        ("teleport", {}, "teleport"),
    ],
)
def test_a_malformed_request_is_told_what_was_wrong(
    method: str, payload: object, echoed: str
) -> None:
    _, _, lane = build_client()
    reply = dispatch_to_wire(lane, method, payload)
    assert reply["ok"] is False
    assert reply["error"] is None
    assert echoed in reply["detail"]


@pytest.mark.parametrize("bad", [None, [], [0.0] * 8, "franka", 3])
def test_an_action_that_is_not_a_map_is_an_invalid_action(bad: object) -> None:
    _, _, lane = build_client()
    ep = lane.reset("stack_cubes").episode_id
    with pytest.raises(GatewayError) as exc:
        InProcessTransport(lane).call("step", {"episode_id": ep, "action": bad})
    assert exc.value.code is ErrorCode.INVALID_ACTION


def test_an_action_that_is_not_a_map_is_an_invalid_action_over_the_socket() -> None:
    from test_http_transport import serve

    from capx.protocol import HttpTransport

    server = serve()
    transport = HttpTransport(server.host, server.port)
    try:
        ep = transport.call("reset", {"task_id": "t"})["episode_id"]
        for bad in (None, [0.0] * 8):
            with pytest.raises(GatewayError) as exc:
                transport.call("step", {"episode_id": ep, "action": bad})
            assert exc.value.code is ErrorCode.INVALID_ACTION
    finally:
        transport.close()
        server.close()
