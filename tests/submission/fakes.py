"""No-sim, no-network test doubles for the submission side of the gateway.

``FakeEnvClient`` implements the methods of :class:`capx.protocol.EnvClient`
(``manifest``/``reset``/``step``/``get_observation``/``finish``/``close``) over
scripted kinematic models (``FakeFrankaModel``, ``FakeBimanualFrankaModel``,
``FakeR1ProModel``), returning the *real*
``ResetResult``/``StepResult``/``FullObs``/``ProprioObs`` dataclasses from
``capx.protocol`` — never redefined here — so a shim under test cannot tell it
apart from a real client at the type level. The shim and runner tests drive it
through the actual shim classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.protocol import (
    ArmProprio,
    BimanualFrankaRawAction,
    CameraObs,
    Embodiment,
    ErrorCode,
    FrankaRawAction,
    FullObs,
    GatewayError,
    ManifestEntry,
    ProprioObs,
    R1ProRawAction,
    RawAction,
    ResetResult,
    StepResult,
)

# ---------------------------------------------------------------------------
# Default manifest — the three taskpacks/original/pack.yaml tasks (task_id,
# embodiment, camera_names kept in sync with that file).
# ---------------------------------------------------------------------------

_DEFAULT_ENTRIES = [
    ManifestEntry(
        task_id="stack_cubes",
        embodiment=Embodiment.FRANKA,
        instruction="Place the primary cube on top of the secondary cube. Quaternions are WXYZ.",
        camera_names=["robot0_robotview"],
        action_spec_ref="PRELIM_RULEBOOK.md#action-franka",
        n_trials=1,
        trial_steplimit=100,
    ),
    ManifestEntry(
        task_id="put_alphabet_soup_and_tomato_sauce_in_basket",
        embodiment=Embodiment.FRANKA,
        instruction=(
            "Put both the alphabet soup and the tomato sauce in the basket. Quaternions are WXYZ."
        ),
        camera_names=["agentview", "robot0_eye_in_hand"],
        action_spec_ref="PRELIM_RULEBOOK.md#action-franka",
        n_trials=1,
        trial_steplimit=100,
    ),
    ManifestEntry(
        task_id="two_arm_lift",
        embodiment=Embodiment.FRANKA_BIMANUAL,
        instruction=(
            "The two arms must coordinate to lift a pot. Arm 0 grasps handle 0 and arm 1 "
            "grasps handle 1, then both lift the pot. Quaternions are WXYZ."
        ),
        camera_names=["agentview"],
        action_spec_ref="PRELIM_RULEBOOK.md#action-franka_bimanual",
        n_trials=1,
        trial_steplimit=100,
    ),
    ManifestEntry(
        task_id="picking_up_trash",
        embodiment=Embodiment.R1PRO,
        instruction="Find the can of soda on the floor and pick it up.",
        camera_names=["ego", "left_wrist", "right_wrist"],
        action_spec_ref="PRELIM_RULEBOOK.md#action-r1pro",
        n_trials=1,
        trial_steplimit=100,
    ),
]


def _default_manifest() -> list[dict[str, Any]]:
    return [entry.model_dump(mode="json") for entry in _DEFAULT_ENTRIES]


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _synthetic_camera(name: str, *, seed: int) -> CameraObs:
    """A deterministic (H, W, 3) RGB + (H, W, 1) depth view, cheap to construct."""
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    depth = np.ones((32, 32, 1), dtype=np.float32)
    return CameraObs(
        name=name,
        rgb=rgb,
        depth=depth,
        intrinsics=np.eye(3, dtype=np.float32),
        pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        pose_mat=np.eye(4, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# FakeFrankaModel
# ---------------------------------------------------------------------------


class FakeFrankaModel:
    """Scripted single-arm Franka kinematics: 8-dim state (7 joints + gripper).

    Per raw cycle: ``joint_pos += clip(target - joint_pos, ±JOINT_STEP)``
    (first-order dynamics toward the commanded target); the gripper tracks its
    commanded fraction the same way. Close enough to the real Robosuite
    adapter's per-cycle contract for shim unit tests, without
    any physics.
    """

    JOINT_STEP = 0.05  # max |Δq| per raw cycle (rad)
    GRIPPER_STEP = 0.1  # max |Δgripper| per raw cycle

    def __init__(self, *, camera_name: str = "robot0_robotview") -> None:
        self.joint_pos = np.zeros(7, dtype=np.float64)
        self.gripper = 1.0
        self.sim_time = 0.0
        self._camera_name = camera_name
        self._camera = _synthetic_camera(camera_name, seed=0)

    def step(self, action: FrankaRawAction) -> None:
        """Advance ``action.repeat`` raw cycles toward ``action.joints``/``action.gripper``."""
        for _ in range(action.repeat):
            self._cycle(action)

    def _cycle(self, action: FrankaRawAction) -> None:
        target = np.asarray(action.joints, dtype=np.float64)
        delta = np.clip(target - self.joint_pos, -self.JOINT_STEP, self.JOINT_STEP)
        self.joint_pos = self.joint_pos + delta
        gripper_delta = float(
            np.clip(action.gripper - self.gripper, -self.GRIPPER_STEP, self.GRIPPER_STEP)
        )
        self.gripper = float(np.clip(self.gripper + gripper_delta, 0.0, 1.0))
        self.sim_time += 1.0

    def proprio_obs(self) -> ProprioObs:
        joint_pos8 = np.concatenate([self.joint_pos, [self.gripper]]).astype(np.float32)
        # Cartesian EEF pose is not kinematically modeled (no FK here); shim
        # tests only assert joint-space convergence, so this is a fixed
        # deterministic placeholder that carries the gripper through, not a
        # real forward-kinematics result.
        cartesian8 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, self.gripper], dtype=np.float32)
        return ProprioObs(joint_pos=joint_pos8, cartesian_pos=cartesian8, sim_time=self.sim_time)

    def full_obs(self) -> FullObs:
        return FullObs(proprio=self.proprio_obs(), cameras={self._camera_name: self._camera})


# ---------------------------------------------------------------------------
# FakeR1ProModel
# ---------------------------------------------------------------------------

_R1PRO_LEFT_ARM_IDX = np.arange(10, 24, 2)  # q28 left-arm slots: 10,12,...,22
_R1PRO_RIGHT_ARM_IDX = np.arange(11, 24, 2)  # q28 right-arm slots: 11,13,...,23


class FakeR1ProModel:
    """Scripted R1Pro kinematics: 28-dim state.

    ``joint_pos`` (28,) layout: base virtual joints ``[0:6]`` (translation
    ``[0:3]``, euler ``[3:6]``; only ``[0:2]`` xy and ``[5]`` yaw are driven
    here), torso ``[6:10]``, interleaved arms ``[10:24]`` (left = even indices
    10,12,...,22; right = odd indices 11,13,...,23), left fingers ``[24:26]``,
    right fingers ``[26:28]``.

    ``action`` is the 25-dim vector ``R1ProKinematics.q28_to_action25`` would
    produce: ``[0:3]`` base-local delta ``[dx, dy, dyaw]``,
    ``[3:7]`` torso absolute target, ``[7:14]``/``[16:23]`` left/right arm
    absolute targets, ``[14:16]``/``[23:25]`` left/right finger absolute
    targets. The base delta is rotated into world frame through the CURRENT
    yaw and applied once per raw cycle — so a held nonzero command drifts
    under high ``repeat``, matching the real controller.
    """

    JOINT_STEP = 0.05
    FINGER_STEP = 0.01

    def __init__(self, *, min_gap: float = 0.0) -> None:
        self.joint_pos = np.zeros(28, dtype=np.float64)
        self.sim_time = 0.0
        # Simulates an object between the fingers that stalls closure once the
        # summed finger gap reaches `min_gap` meters, even if the command
        # targets full closure. 0.0 disables the obstruction (fingers close
        # fully, tracking the command like any other joint).
        self._min_gap = min_gap

    def step(self, action: R1ProRawAction) -> None:
        """Advance ``action.repeat`` raw cycles per the 25-dim ``action`` vector."""
        a = np.asarray(action.action, dtype=np.float64)
        for _ in range(action.repeat):
            self._cycle(a)

    def _cycle(self, a: np.ndarray) -> None:
        yaw = self.joint_pos[5]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        self.joint_pos[0:2] = self.joint_pos[0:2] + rot @ a[0:2]
        self.joint_pos[5] = _wrap_to_pi(self.joint_pos[5] + a[2])

        self.joint_pos[6:10] = self._track(self.joint_pos[6:10], a[3:7], self.JOINT_STEP)
        self.joint_pos[_R1PRO_LEFT_ARM_IDX] = self._track(
            self.joint_pos[_R1PRO_LEFT_ARM_IDX], a[7:14], self.JOINT_STEP
        )
        self.joint_pos[_R1PRO_RIGHT_ARM_IDX] = self._track(
            self.joint_pos[_R1PRO_RIGHT_ARM_IDX], a[16:23], self.JOINT_STEP
        )
        self.joint_pos[24:26] = self._track_fingers(self.joint_pos[24:26], a[14:16])
        self.joint_pos[26:28] = self._track_fingers(self.joint_pos[26:28], a[23:25])
        self.sim_time += 1.0

    @staticmethod
    def _track(current: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
        delta = np.clip(target - current, -step, step)
        return current + delta

    def _track_fingers(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        tracked = self._track(current, target, self.FINGER_STEP)
        gap = float(np.sum(tracked))
        if self._min_gap > 0.0 and gap < self._min_gap:
            # Obstruction: split the shortfall evenly across both fingers so
            # the summed gap stalls at `_min_gap` instead of closing fully.
            tracked = tracked + (self._min_gap - gap) / 2.0
        return tracked

    def proprio_obs(self) -> ProprioObs:
        cartesian_pos = np.zeros(14, dtype=np.float32)  # not kinematically modeled, see class doc
        return ProprioObs(
            joint_pos=self.joint_pos.astype(np.float32),
            cartesian_pos=cartesian_pos,
            sim_time=self.sim_time,
        )

    def full_obs(self) -> FullObs:
        cameras = {
            name: _synthetic_camera(name, seed=i)
            for i, name in enumerate(("ego", "left_wrist", "right_wrist"))
        }
        return FullObs(proprio=self.proprio_obs(), cameras=cameras)


class FakeBimanualFrankaModel:
    """Two ``FakeFrankaModel`` arms stepped together, as one two-arm control cycle is.

    Arm 0's base is the world origin and arm 1's sits 1 m along y; the proprio
    mirrors arm 0 at the top level, as the gateway's does.
    """

    _BASE_POSES = (
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    def __init__(self, *, camera_name: str = "agentview") -> None:
        self.arms = [FakeFrankaModel(camera_name=camera_name) for _ in self._BASE_POSES]
        self._camera_name = camera_name

    def step(self, action: BimanualFrankaRawAction) -> None:
        joints = np.asarray(action.joints, dtype=np.float64).reshape(2, 7)
        grippers = np.asarray(action.grippers, dtype=np.float64).reshape(2)
        for arm, model in enumerate(self.arms):
            model.step(FrankaRawAction(joints=joints[arm], gripper=float(grippers[arm]), repeat=action.repeat))

    def proprio_obs(self) -> ProprioObs:
        per_arm = [model.proprio_obs() for model in self.arms]
        return ProprioObs(
            joint_pos=per_arm[0].joint_pos,
            cartesian_pos=per_arm[0].cartesian_pos,
            sim_time=per_arm[0].sim_time,
            arms=[
                ArmProprio(joint_pos=obs.joint_pos, cartesian_pos=obs.cartesian_pos, base_pose=base)
                for obs, base in zip(per_arm, self._BASE_POSES, strict=True)
            ],
        )

    def full_obs(self) -> FullObs:
        camera = self.arms[0].full_obs().cameras[self._camera_name]
        return FullObs(proprio=self.proprio_obs(), cameras={self._camera_name: camera})


_FakeModel = FakeFrankaModel | FakeBimanualFrankaModel | FakeR1ProModel


# ---------------------------------------------------------------------------
# FakeEnvClient
# ---------------------------------------------------------------------------


class FakeEnvClient:
    """No-sim, no-network stand-in for :class:`capx.protocol.EnvClient`.

    Drives one fake model, matched to the task's embodiment, per active episode from
    the real ``RawAction``/``ResetResult``/``StepResult``/``FullObs``/
    ``ProprioObs`` dataclasses. Supports the fault injection the shim/runner
    unit tests need: a call-indexed error injection, flipping
    ``terminated``/``truncated`` after N steps of an episode, and a per-task
    trial budget that raises ``TRIALS_EXHAUSTED`` on an over-budget reset.

    Every action passed to :meth:`step` is recorded (in call order) on
    ``self.actions`` so a test can assert exactly what was sent over the wire.
    """

    def __init__(self, manifest: list[dict[str, Any]] | None = None) -> None:
        self.manifest_entries = manifest if manifest is not None else _default_manifest()
        self.actions: list[RawAction] = []
        self._models: dict[str, _FakeModel] = {}
        self._steps_since_reset: dict[str, int] = {}
        self._flags: dict[str, tuple[bool, bool]] = {}
        self._flip_after: tuple[int, bool, bool] | None = None
        self._call_count = 0
        self._fail_at: dict[int, GatewayError] = {}
        self._trial_budget: dict[str, int] = {}
        self._trials_used: dict[str, int] = {}
        self._min_gap: dict[str, float] = {}
        self._next_episode = 0
        self.finished = False

    # -- fault injection ----------------------------------------------------

    def fail_at(self, call_index: int, error: GatewayError) -> None:
        """Raise ``error`` on the ``call_index``-th call (0-based) to reset/step/get_observation."""
        self._fail_at[call_index] = error

    def set_flags_after(self, n_steps: int, *, terminated: bool = False, truncated: bool = False) -> None:
        """From the ``n_steps``-th ``step`` of any episode onward, report these flags."""
        self._flip_after = (n_steps, terminated, truncated)

    def trial_budget(self, task_id: str, n: int) -> None:
        """Cap ``task_id`` to ``n`` resets; the ``(n + 1)``-th raises ``TRIALS_EXHAUSTED``."""
        self._trial_budget[task_id] = n

    def set_min_gap(self, task_id: str, min_gap: float) -> None:
        """Simulate an object stalling gripper closure on r1pro episodes of ``task_id``."""
        self._min_gap[task_id] = min_gap

    # -- EnvProtocol-shaped surface ------------------------------------------

    def manifest(self) -> list[dict[str, Any]]:
        return self.manifest_entries

    def reset(self, task_id: str) -> ResetResult:
        self._maybe_fail()
        used = self._trials_used.get(task_id, 0)
        budget = self._trial_budget.get(task_id)
        if budget is not None and used >= budget:
            raise GatewayError(ErrorCode.TRIALS_EXHAUSTED, f"no trials left for task {task_id!r}")
        self._trials_used[task_id] = used + 1

        entry = self._entry(task_id)
        episode_id = f"ep-{self._next_episode}"
        self._next_episode += 1
        model = self._build_model(entry)
        self._models[episode_id] = model
        self._steps_since_reset[episode_id] = 0
        self._flags[episode_id] = (False, False)
        return ResetResult(
            episode_id=episode_id, obs=model.full_obs(), instruction=str(entry["instruction"])
        )

    def step(self, episode_id: str, action: RawAction) -> StepResult:
        self._maybe_fail()
        model = self._require(episode_id)
        self.actions.append(action)
        model.step(action)  # type: ignore[arg-type]  # matched to embodiment at reset time
        self._steps_since_reset[episode_id] += 1

        terminated, truncated = self._flags[episode_id]
        if self._flip_after is not None:
            n_steps, flip_terminated, flip_truncated = self._flip_after
            if self._steps_since_reset[episode_id] >= n_steps:
                terminated, truncated = flip_terminated, flip_truncated
                self._flags[episode_id] = (terminated, truncated)

        obs: FullObs | ProprioObs = model.full_obs() if action.include_cameras else model.proprio_obs()
        return StepResult(obs=obs, terminated=terminated, truncated=truncated)

    def get_observation(self, episode_id: str) -> FullObs:
        self._maybe_fail()
        return self._require(episode_id).full_obs()

    def finish(self) -> None:
        self.finished = True

    def close(self) -> None:
        self.closed = True
        self._models.clear()

    # -- internals ------------------------------------------------------------

    def _maybe_fail(self) -> None:
        error = self._fail_at.pop(self._call_count, None)
        self._call_count += 1
        if error is not None:
            raise error

    def _entry(self, task_id: str) -> dict[str, Any]:
        for entry in self.manifest_entries:
            if entry["task_id"] == task_id:
                return entry
        raise ValueError(f"unknown task_id: {task_id!r}")

    def _build_model(self, entry: dict[str, Any]) -> _FakeModel:
        camera_names = entry["camera_names"]
        if entry["embodiment"] == Embodiment.FRANKA.value:
            return FakeFrankaModel(camera_name=camera_names[0] if camera_names else "robot0_robotview")
        if entry["embodiment"] == Embodiment.FRANKA_BIMANUAL.value:
            return FakeBimanualFrankaModel(camera_name=camera_names[0] if camera_names else "agentview")
        min_gap = self._min_gap.get(str(entry["task_id"]), 0.0)
        return FakeR1ProModel(min_gap=min_gap)

    def _require(self, episode_id: str) -> _FakeModel:
        model = self._models.get(episode_id)
        if model is None:
            raise GatewayError(ErrorCode.CLIENT_TIMEOUT, f"unknown episode {episode_id!r}")
        return model


__all__ = ["FakeBimanualFrankaModel", "FakeEnvClient", "FakeFrankaModel", "FakeR1ProModel"]

