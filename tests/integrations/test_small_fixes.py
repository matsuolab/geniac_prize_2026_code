"""Regression tests for small fixes in capx/integrations.

The modules import perception and IK stacks the unit-test venv does not have, so
each test loads its module from the file with stand-ins for those imports. The
stand-ins are removed from ``sys.modules`` again after the test.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Rotation:
    """Enough of scipy's Rotation for the code under test (unit quaternions)."""

    def __init__(self, xyzw: np.ndarray) -> None:
        self.xyzw = np.asarray(xyzw, dtype=np.float64) / np.linalg.norm(xyzw)

    @classmethod
    def from_quat(cls, quat: Any, scalar_first: bool = False) -> _Rotation:
        quat = np.asarray(quat, dtype=np.float64)
        return cls(np.roll(quat, -1) if scalar_first else quat)

    def apply(self, vector: Any) -> np.ndarray:
        x, y, z, w = self.xyzw
        matrix = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return matrix @ np.asarray(vector, dtype=np.float64)


SCIPY = {
    "scipy": {},
    "scipy.spatial": {"ConvexHull": object},
    "scipy.spatial.transform": {"Rotation": _Rotation},
}
CAPX_STACK = {
    "open3d": {},
    "viser": {"ViserServer": object},
    "viser.transforms": {},
    "viser.extras": {"ViserUrdf": object},
    "matplotlib": {},
    "matplotlib.pyplot": {},
    "matplotlib.patches": {},
    "requests": {"RequestException": Exception},
    "capx.utils.unix_http": {"check_server_url": lambda url: url},
    "capx.utils.unix_http_client": {"post": None, "new_session": None},
    "capx.envs": {},
    "capx.envs.base": {"BaseEnv": object},
    "capx.integrations": {},
    "capx.integrations.base_api": {"ApiBase": object},
    "capx.integrations.franka": {},
    "capx.integrations.franka.common": {
        name: None for name in ("apply_tcp_offset", "close_gripper", "open_gripper")
    },
    "capx.integrations.motion": {},
    "capx.integrations.motion.pyroki": {"init_pyroki": None},
    "capx.integrations.motion.pyroki_snippets": {},
    "capx.integrations.motion.pyroki_context": {"get_pyroki_context": None},
    "capx.integrations.vision": {},
    "capx.integrations.vision.graspnet": {"init_contact_graspnet": None},
    "capx.integrations.vision.owlvit": {"init_owlvit": None},
    "capx.integrations.vision.molmo": {"init_molmo": None},
    "capx.integrations.vision.sam2": {"init_sam2": None, "init_sam2_point_prompt": None},
    "capx.integrations.vision.sam3": {
        name: None for name in ("init_sam3", "init_sam3_point_prompt", "visualize_sam3_results")
    },
    "capx.integrations.r1pro": {},
    "capx.integrations.r1pro.utils": {},
    "capx.utils.depth_utils": {
        name: None
        for name in (
            "deproject_pixel_to_camera",
            "depth_color_to_pointcloud",
            "depth_to_pointcloud",
            "depth_to_rgb",
        )
    },
    "capx.utils.video_utils": {"_write_video": None},
    "torch": {},
    "torchvision": {},
    "torchvision.transforms": {},
    "torchvision.transforms.functional": {"to_pil_image": None},
    "jax": {},
    "jax.numpy": {},
}


@pytest.fixture
def load(monkeypatch: pytest.MonkeyPatch):
    def load(relpath: str, **overrides: dict[str, Any]) -> types.ModuleType:
        for name, attrs in {**SCIPY, **CAPX_STACK, **overrides}.items():
            stub = types.ModuleType(name)
            stub.__path__ = []
            stub.__dict__.update(attrs)
            monkeypatch.setitem(sys.modules, name, stub)
            # `from package import name` reads a loaded package's attribute first.
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                monkeypatch.setattr(parent, child, stub, raising=False)
        path = REPO_ROOT / relpath
        spec = importlib.util.spec_from_file_location(f"_under_test_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


def _rows(r0: int, r1: int) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    mask[r0:r1] = True
    return mask


# ----------------------------- SAM2 / SAM3 mask choice -----------------------------


@pytest.mark.parametrize("best_first", [True, False])
def test_box_prompt_keeps_the_highest_scoring_sam2_mask(load, best_first: bool) -> None:
    common = load("capx/integrations/franka/common.py")
    good, bad = {"mask": _rows(0, 2), "score": 0.9}, {"mask": _rows(4, 8), "score": 0.2}
    masks = [good, bad] if best_first else [bad, good]
    seg = common.build_segmentation_map_from_sam2(
        lambda rgb, box=None: masks, np.zeros((8, 8, 3), np.uint8), {}, box=[0, 0, 8, 8]
    )
    np.testing.assert_array_equal(seg[:, :, 0] > 0, good["mask"])


def test_unscored_sam2_masks_are_all_kept(load) -> None:
    common = load("capx/integrations/franka/common.py")
    masks = [{"mask": _rows(0, 2)}, {"mask": _rows(4, 8)}]
    seg = common.build_segmentation_map_from_sam2(
        lambda rgb, box=None: masks, np.zeros((8, 8, 3), np.uint8), {}, box=[0, 0, 8, 8]
    )
    assert set(np.unique(seg)) == {0, 1, 2}


def test_nut_assembly_uses_the_highest_scoring_sam3_mask(load) -> None:
    module = load("capx/integrations/franka/nut_assembly_visual.py")
    api = module.FrankaControlNutAssemblyVisualApi.__new__(module.FrankaControlNutAssemblyVisualApi)
    api.molmo_point_fn = lambda image, objects: {objects[0]: (3, 5)}
    api.sam3_point_prompt_fn = lambda image, point: [
        {"mask": _rows(0, 2), "score": 0.2},
        {"mask": _rows(4, 8), "score": 0.9},
    ]
    mask, point, scores = api._segment_object_from_language(Image.new("RGB", (8, 8)), "nut")
    np.testing.assert_array_equal(mask, _rows(4, 8))
    assert point == (3, 5) and scores == [0.2, 0.9]


def test_nut_assembly_reports_an_empty_sam3_result(load) -> None:
    module = load("capx/integrations/franka/nut_assembly_visual.py")
    api = module.FrankaControlNutAssemblyVisualApi.__new__(module.FrankaControlNutAssemblyVisualApi)
    api.molmo_point_fn = lambda image, objects: {objects[0]: (3, 5)}
    api.sam3_point_prompt_fn = lambda image, point: []
    with pytest.raises(ValueError, match="no masks"):
        api._segment_object_from_language(Image.new("RGB", (8, 8)), "nut")


# ----------------------------- Molmo -----------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ('<points coords="1 1 500 250">mug</points>', [(500.0, 250.0)]),
        ('<points coords="1 1 500 250 2 100">mug</points>', [(500.0, 250.0)]),
        ('<points coords="0 0 100">mug</points>', []),
        ('<points coords="1 1 -40 250 2 1500 20">mug</points>', []),
        ('<points coords="1 1 a b">mug</points>', []),
    ],
)
def test_molmo2_coordinates_need_complete_in_range_triplets(load, text, expected) -> None:
    molmo = load("capx/integrations/vision/molmo.py")
    assert molmo._parse_points(text) == (expected, 1000.0)


def test_molmo_plain_pairs_keep_their_sign(load) -> None:
    molmo = load("capx/integrations/vision/molmo.py")
    assert molmo._parse_points("The mug is at (-40, 30).")[0] == []
    assert molmo._parse_points("The mug is at (40, 30).")[0] == [(40.0, 30.0)]


# ----------------------------- sidecar clients -----------------------------


def test_molmo_does_not_retry_through_a_deadline(load) -> None:
    attempts = []

    class Session:
        def post(self, *args: Any, **kwargs: Any) -> None:
            attempts.append(1)
            raise TimeoutError("trial time is up")

    molmo = load(
        "capx/integrations/vision/molmo.py",
        **{"capx.utils.unix_http_client": {"new_session": Session}},
    )
    with pytest.raises(TimeoutError):
        molmo.init_molmo()(Image.new("RGB", (8, 8)), objects=["mug"])
    assert len(attempts) == 1


def _post_with_retries(behaviour):
    return {"capx.utils.serve_utils": {"post_with_retries": behaviour}}


def test_sam3_service_failure_is_an_error_not_an_empty_result(load) -> None:
    def down(url: str, payload: dict, **kwargs: Any) -> None:
        raise RuntimeError("Request to http+unix://sam3.sock/segment failed after 5 retries")

    sam3 = load("capx/integrations/vision/sam3.py", **_post_with_retries(down))
    with pytest.raises(RuntimeError, match="failed after 5 retries"):
        sam3.init_sam3()(np.zeros((4, 4, 3), np.uint8), text_prompt="mug")


def test_sam3_deadline_is_not_swallowed(load) -> None:
    def deadline(url: str, payload: dict, **kwargs: Any) -> None:
        raise TimeoutError("trial time is up")

    sam3 = load("capx/integrations/vision/sam3.py", **_post_with_retries(deadline))
    with pytest.raises(TimeoutError):
        sam3.init_sam3()(np.zeros((4, 4, 3), np.uint8), text_prompt="mug")


def test_sam3_no_detection_is_still_an_empty_list(load) -> None:
    sam3 = load(
        "capx/integrations/vision/sam3.py", **_post_with_retries(lambda *a, **k: {"results": []})
    )
    assert sam3.init_sam3()(np.zeros((4, 4, 3), np.uint8), text_prompt="mug") == []


def test_graspnet_requests_have_a_timeout(load) -> None:
    seen: list[dict[str, Any]] = []
    empty = None

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"grasps_base64": empty, "scores_base64": empty, "contact_pts_base64": empty}

    def post(url: str, **kwargs: Any) -> Response:
        seen.append(kwargs)
        return Response()

    graspnet = load(
        "capx/integrations/vision/graspnet.py",
        **{"capx.utils.unix_http_client": {"post": post}},
    )
    empty = graspnet._numpy_to_base64(np.zeros((0, 4, 4)))
    graspnet.init_contact_graspnet()(np.zeros((4, 4)), np.eye(3), np.zeros((4, 4)), 1)
    graspnet.init_contact_graspnet_point_clouds()(np.zeros((3, 3)), np.zeros((3, 3)))
    assert [kwargs["timeout"] for kwargs in seen] == [graspnet.REQUEST_TIMEOUT_S] * 2


# ----------------------------- Franka privileged APIs -----------------------------


def test_privileged_cube_pose_has_no_extent_unless_asked(load) -> None:
    module = load("capx/integrations/franka/control_privileged.py")
    api = module.FrankaControlPrivilegedApi.__new__(module.FrankaControlPrivilegedApi)
    cube = np.array([0.5, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])
    api._env = types.SimpleNamespace(
        get_observation=lambda: {"cube_poses": {"primary": cube, "secondary": cube}}
    )
    assert api.get_object_pose("red cube")[2] is None
    np.testing.assert_allclose(api.get_object_pose("green cube", True)[2], [0.05, 0.05, 0.05])


class _InteractiveEnv:
    _control_freq = 20

    def __init__(self, cartesian: np.ndarray) -> None:
        self.cartesian = cartesian
        self.time_s = 0.0
        self.moves = 0

    def get_observation(self) -> dict[str, np.ndarray]:
        return {"robot_cartesian_pos": self.cartesian}

    def get_current_time_s(self) -> float:
        self.time_s += 0.05
        return self.time_s

    def move_to_joints_blocking(self, joints: np.ndarray, **kwargs: Any) -> None:
        self.moves += 1


def _interactive_api(load, cartesian: np.ndarray):
    module = load("capx/integrations/franka/libero_privileged.py")
    api = module.FrankaLiberoPrivilegedApi.__new__(module.FrankaLiberoPrivilegedApi)
    targets: list[np.ndarray] = []

    def solve_ik_vel_cost(target_position, target_wxyz, **kwargs: Any) -> np.ndarray:
        targets.append(np.array(target_position))
        # The arm tracks each step exactly.
        api._env.cartesian = np.concatenate([target_position, target_wxyz, [1.0]])
        return np.zeros(8)

    api._env = _InteractiveEnv(cartesian)
    api._pks = types.SimpleNamespace(solve_ik_vel_cost=solve_ik_vel_cost)
    api._robot, api._target_link_name, api.cfg = None, "panda_hand", None
    return api, targets


def test_interactive_move_ends_when_the_target_is_the_same_rotation_as_minus_q(load) -> None:
    down = np.array([0.0, 1.0, 0.0, 0.0])
    hand = np.array([0.5, 0.0, 0.3])
    api, targets = _interactive_api(load, np.concatenate([hand, -down, [1.0]]))
    tcp = hand + np.array([0.0, 0.0, -0.1])  # the hand sits 0.1 m above a downward TCP

    api.goto_pose_interactive_cartesian(lambda: (tcp, down), timeout_s=5.0)

    assert len(targets) <= 1


def test_interactive_replanning_keeps_the_tcp_offset(load) -> None:
    down = np.array([0.0, 1.0, 0.0, 0.0])
    api, targets = _interactive_api(load, np.array([0.5, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0, 1.0]))
    tcp = np.array([0.5, 0.05, 0.2])

    api.goto_pose_interactive_cartesian(
        lambda: (tcp, down), replan_interval_s=0.01, lin_vel_norm=0.5, timeout_s=5.0
    )

    np.testing.assert_allclose(targets[-1], tcp + [0.0, 0.0, 0.1], atol=1e-9)


# ----------------------------- R1Pro -----------------------------


def _r1pro(load, start: tuple[float, float, float], reachable):
    module = load("capx/integrations/r1pro/control.py")
    api = module.R1ProControlApi.__new__(module.R1ProControlApi)
    visited: list[np.ndarray] = []

    def navigate(pose: np.ndarray) -> bool:
        visited.append(np.array(pose))
        return reachable(np.array(pose))

    api._env = types.SimpleNamespace(_navigate_to_pose=navigate)
    api.get_robot_position = lambda: (np.array([start[0], start[1], 0.0]), None, start[2])
    return api, visited


def test_navigation_reports_failure_when_only_a_waypoint_was_reached(load) -> None:
    api, visited = _r1pro(load, (0.0, 0.0, 0.0), lambda pose: pose[0] <= 1.0)
    assert api.navigate_to_pose((2.0, 0.0, 0.0)) is False
    np.testing.assert_allclose(visited[-1], [1.0, 0.0, 0.0])


def test_navigation_never_falls_back_to_the_start_pose(load) -> None:
    api, visited = _r1pro(load, (0.0, 0.0, 0.0), lambda pose: False)
    assert api.navigate_to_pose((2.0, 0.0, 1.0)) is False
    assert all(pose[0] > 0.0 for pose in visited)


def test_navigation_turns_the_short_way_across_pi(load) -> None:
    api, visited = _r1pro(load, (0.0, 0.0, 3.0), lambda pose: False)
    api.navigate_to_pose((2.0, 0.0, -3.0))
    halfway = next(pose for pose in visited if np.isclose(pose[0], 1.0))
    assert abs(halfway[2]) == pytest.approx(np.pi, abs=1e-9)


def test_navigation_succeeds_when_the_goal_is_reached(load) -> None:
    api, visited = _r1pro(load, (0.0, 0.0, 0.0), lambda pose: True)
    assert api.navigate_to_pose((2.0, 1.0, 0.5)) is True
    np.testing.assert_allclose(visited, [[2.0, 1.0, 0.5]])
