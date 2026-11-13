"""Scripted (no-LLM) single-grasp probe for the BEHAVIOR/R1Pro gateway path.

Reproduces the exact call sequence the measured LLM agent used on
`picking_up_trash` (find -> navigate -> sample_grasp_pose -> grasp_object ->
check_object_in_hand), with heavy instrumentation:

- every `_remote_step` RPC is tagged with the primitive that issued it and
  logged (wall time, repeat, include_cameras, finger gap, base pose) to
  `rpc_log.jsonl`;
- every pose handoff in the grasp pipeline (camera pose, object OBB, grasp
  candidates, IK target in world/base frame, IK seed vs solution, post-move
  FK-vs-wire cross-check) is logged to `events.jsonl`;
- ego + wrist camera RGB frames are saved at each milestone.

Run (host, .venv-gwtest, backend on behavior.sock, SAM3 on sam3.sock, GraspNet on graspnet.sock):

    PYTHONPATH=. JAX_PLATFORMS=cpu .venv-gwtest/bin/python \
        scripts/behavior_grasp_probe.py --out outputs/behavior_debug/probe_01

Isaac boots on the first reset RPC (~5-8 min warm) — do not add client
timeouts. One backend process serves ONE gateway connection; `docker restart
behavior-backend` between runs.
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import time
from typing import Any

import numpy as np

_T0 = time.monotonic()


class ProbeLog:
    """JSONL event/RPC logger with a mutable phase tag."""

    def __init__(self, out_dir: pathlib.Path) -> None:
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self._rpc_f = open(out_dir / "rpc_log.jsonl", "w")
        self._ev_f = open(out_dir / "events.jsonl", "w")
        self.phase = "init"

    @staticmethod
    def _clean(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {k: ProbeLog._clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ProbeLog._clean(v) for v in value]
        if hasattr(value, "detach"):  # torch tensor
            return value.detach().cpu().numpy().tolist()
        return value

    def rpc(self, **fields: Any) -> None:
        fields["t"] = round(time.monotonic() - _T0, 3)
        fields["phase"] = self.phase
        self._rpc_f.write(json.dumps(self._clean(fields)) + "\n")
        self._rpc_f.flush()

    def event(self, kind: str, **fields: Any) -> None:
        fields["t"] = round(time.monotonic() - _T0, 3)
        fields["kind"] = kind
        fields["phase"] = self.phase
        self._ev_f.write(json.dumps(self._clean(fields)) + "\n")
        self._ev_f.flush()
        print(f"[{fields['t']:9.1f}s] {kind}: " + json.dumps(self._clean(fields), default=str)[:400])


def save_cams(shim: Any, out_dir: pathlib.Path, tag: str, log: ProbeLog) -> None:
    """Save ego + wrist RGB (and ego depth) from the latest full obs."""
    from PIL import Image

    obs = shim._fetch_full_obs()
    for name in ("ego", "left_wrist"):
        cam = obs.cameras.get(name)
        if cam is None:
            continue
        Image.fromarray(np.asarray(cam.rgb, dtype=np.uint8)).save(out_dir / f"{tag}_{name}.png")
        if name == "ego" and cam.depth is not None:
            np.save(out_dir / f"{tag}_ego_depth.npy", np.asarray(cam.depth))
    log.event("saved_cams", tag=tag)


def instrument_remote_step(shim: Any, log: ProbeLog) -> None:
    orig = shim._remote_step

    @functools.wraps(orig)
    def wrapped(action: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        result = orig(action, **kwargs)
        dt = time.monotonic() - t0
        q = shim._q28_now()
        has_nan = bool(np.any(np.isnan(q)))
        log.rpc(
            dt=round(dt, 4),
            repeat=int(action.repeat),
            include_cameras=bool(action.include_cameras),
            base=[round(float(v), 5) for v in q[0:6]],
            lgap=round(float(q[24] + q[25]), 5),
            act_base=[round(float(v), 5) for v in np.asarray(action.action)[0:3]],
            nan=has_nan,
        )
        if has_nan:
            log.event("PROPRIO_NAN", q28=q)
        return result

    shim._remote_step = wrapped

    orig_get_obs = shim._client.get_observation

    @functools.wraps(orig_get_obs)
    def wrapped_obs(episode_id: str) -> Any:
        t0 = time.monotonic()
        result = orig_get_obs(episode_id)
        log.rpc(dt=round(time.monotonic() - t0, 4), repeat=0, include_cameras=True, rpc="get_observation")
        return result

    shim._client.get_observation = wrapped_obs


def snapshot_state(shim: Any, log: ProbeLog, tag: str) -> None:
    """Cross-check client FK vs the wire's server-side base-frame eef pose."""
    proprio = shim._latest_proprio()
    q28 = shim._q28_now()
    fk_pos, fk_quat_xyzw = shim._fk_eef_base(0)
    wire = np.asarray(proprio.cartesian_pos, dtype=np.float64)
    wire_left_pos, wire_left_wxyz = wire[0:3], wire[3:7]
    base_pos, base_quat = shim.robot.get_position_orientation()
    log.event(
        "state_snapshot",
        tag=tag,
        q28_base=q28[0:6],
        fk_left_pos_base=fk_pos,
        fk_left_quat_xyzw_base=fk_quat_xyzw,
        wire_left_pos_base=wire_left_pos,
        wire_left_quat_wxyz_base=wire_left_wxyz,
        fk_minus_wire_pos=fk_pos - wire_left_pos,
        base_view_pos=base_pos,
        base_view_quat_xyzw=base_quat,
        lgap=float(q28[24] + q28[25]),
        rgap=float(q28[26] + q28[27]),
    )


def instrumented_grasp(api: Any, shim: Any, log: ProbeLog, out_dir: pathlib.Path,
                       pregrasp_pose: Any, grasp_pose: Any, tag: str, arm: int = 0) -> bool:
    """Mirror of R1ProControlApi.grasp_object with pose/IK logging at each hop."""
    log.phase = f"grasp:{tag}"
    log.event("grasp_begin", tag=tag,
              pregrasp_pos=pregrasp_pose[0], pregrasp_quat_xyzw=pregrasp_pose[1],
              grasp_pos=grasp_pose[0], grasp_quat_xyzw=grasp_pose[1])

    api.open_gripper(arm=arm)
    snapshot_state(shim, log, f"{tag}:after_open")

    target_pos_world = grasp_pose[0] + np.array([0.0, 0.0, -0.02])
    target_quat_wxyz = grasp_pose[1][[3, 0, 1, 2]]

    base_pos, base_quat = shim.robot.get_position_orientation()
    seed_q28 = shim._q28_now()
    joints = api.solve_ik(target_pos_world, target_quat_wxyz, arm=arm)
    delta = np.asarray(joints, dtype=np.float64) - seed_q28
    log.event(
        "ik_solved", tag=tag,
        target_pos_world=target_pos_world, target_quat_wxyz_world=target_quat_wxyz,
        base_pos=np.asarray(base_pos), base_quat_xyzw=np.asarray(base_quat),
        seed_q28=seed_q28, solution_q28=joints,
        max_abs_dq=float(np.max(np.abs(delta[6:24]))),
        dq_torso=delta[6:10], dq_left=delta[10:24:2],
        ik_residual=api.last_ik_pos_residual,
    )
    # mirror grasp_object's reachability guard so probe behavior == API behavior
    if api.last_ik_pos_residual is not None and api.last_ik_pos_residual > api._ik_residual_limit:
        log.event("grasp_refused_unreachable", tag=tag, residual=api.last_ik_pos_residual)
        return False

    t0 = time.monotonic()
    success = api.move_to_joint_positions(joints, max_steps=200, settle_steps=5)
    shim._settle_robot()
    log.event("move_done", tag=tag, servo_converged=bool(success),
              dt=round(time.monotonic() - t0, 1),
              residual_q=(shim._q28_now() - np.asarray(joints))[6:24])
    save_cams(shim, out_dir, f"{tag}_post_move", log)

    current_eef = api.get_current_eef_pose(arm=arm)
    eef_err = np.asarray(current_eef[0], dtype=np.float64) - np.asarray(grasp_pose[0], dtype=np.float64)
    log.event("eef_after_move", tag=tag, eef_pos_world=current_eef[0],
              eef_quat_xyzw_world=current_eef[1], err_vs_grasp=eef_err,
              err_norm=float(np.linalg.norm(eef_err)))
    snapshot_state(shim, log, f"{tag}:after_move")

    if np.linalg.norm(eef_err) > 0.03:
        diff = np.asarray(current_eef[0], dtype=np.float64) - np.asarray(grasp_pose[0], dtype=np.float64)
        diff[-1] = 0.08
        new_grasp_pos = grasp_pose[0] - diff
        log.event("correction_move", tag=tag, new_grasp_pos=new_grasp_pos)
        joints = api.solve_ik(new_grasp_pos, target_quat_wxyz, arm=arm)
        success = api.move_to_joint_positions(joints, max_steps=200, settle_steps=5)
        shim._settle_robot()
        current_eef = api.get_current_eef_pose(arm=arm)
        eef_err = np.asarray(current_eef[0], dtype=np.float64) - np.asarray(grasp_pose[0], dtype=np.float64)
        log.event("eef_after_correction", tag=tag, eef_pos_world=current_eef[0],
                  err_vs_grasp=eef_err, err_norm=float(np.linalg.norm(eef_err)))
        save_cams(shim, out_dir, f"{tag}_post_correction", log)

    log.event("closing_gripper", tag=tag, lgap_before=float(shim._q28_now()[24] + shim._q28_now()[25]))
    api.close_gripper(arm=arm)
    q = shim._q28_now()
    log.event("closed_gripper", tag=tag, lgap_after=float(q[24] + q[25]), fingers=q[24:28])
    save_cams(shim, out_dir, f"{tag}_post_close", log)

    shim._settle_robot()
    q = shim._q28_now()
    log.event("post_settle_gap", tag=tag, lgap=float(q[24] + q[25]))

    in_hand_before_lift = shim.check_object_in_hand(arm=arm)
    log.event("in_hand_before_lift", tag=tag, value=bool(in_hand_before_lift),
              lgap=float(shim._q28_now()[24] + shim._q28_now()[25]))

    api.lift_arm(arm=arm)
    api.reset_torso()
    save_cams(shim, out_dir, f"{tag}_post_lift", log)

    in_hand = shim.check_object_in_hand(arm=arm)
    q = shim._q28_now()
    log.event("grasp_end", tag=tag, in_hand=bool(in_hand), lgap=float(q[24] + q[25]), fingers=q[24:28])
    return bool(in_hand)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/behavior_debug/probe_01")
    parser.add_argument("--backend", default="behavior=127.0.0.1:18102")
    parser.add_argument("--taskpack", default="taskpacks/original")
    parser.add_argument("--task-id", default="picking_up_trash")
    parser.add_argument("--object", default="can of soda")
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="grasp candidate pairs to try (agent tried all, default 2 = one pass)")
    parser.add_argument("--skip-nav", action="store_true",
                        help="grasp from wherever the robot is (no navigate)")
    parser.add_argument("--approach-until-reachable", action="store_true",
                        help="repeat detect->navigate hops until the freshest detection is "
                        "within --reach-radius of the base, then grasp that one")
    parser.add_argument("--reach-radius", type=float, default=0.55)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--nan-replay", type=int, default=0, metavar="N",
                        help="skip perception; instead replay N aggressive far-target "
                        "_move_hand/IK motions (goal-2 physics-NaN reproduction) and exit")
    parser.add_argument("--route", default="",
                        help="semicolon-separated x,y waypoints (root frame) to drive "
                        "BEFORE the final detect->standoff->grasp (works around the "
                        "obstacle-blind straight-line nav; e.g. '-0.35,-0.99;0.62,0.85')")
    parser.add_argument("--extra", default="",
                        help="comma-separated key=value R1ProKnobs overrides, e.g. "
                        "'settle_cycles=10,nav_max_lin=0.15,max_joint_delta_per_rpc=0.5'")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    log = ProbeLog(out_dir)

    from capx.gateway import EnforcementMode, Gateway
    from capx.gateway.env_backend import RemoteEnvFactory, parse_backend_routes
    from capx.integrations.r1pro.control import R1ProControlApi
    from capx.protocol import EnvClient, InProcessTransport
    from capx.protocol.taskpack import load_task_packs
    from capx.submission.interfaces import R1ProKnobs
    from capx.submission.shims.r1pro import R1ProGatewayShim

    tasks = load_task_packs([args.taskpack])
    spec = tasks[args.task_id]
    routes = parse_backend_routes([e for e in args.backend.split(",") if e.strip()])

    with Gateway(tasks, RemoteEnvFactory(routes), EnforcementMode.DEV) as gateway:
        client = EnvClient(InProcessTransport(gateway.acquire()))
        overrides: dict[str, float] = {}
        for kv in (s for s in args.extra.split(",") if s.strip()):
            key, _, value = kv.partition("=")
            overrides[key.strip()] = float(value)
        shim = R1ProGatewayShim(
            client, args.task_id, camera_names=list(spec.camera_names),
            step_chunk=10, knobs=R1ProKnobs.model_validate(overrides),
        )
        instrument_remote_step(shim, log)

        log.event("reset_begin", note="Isaac boot on first reset takes ~5-8 min warm")
        obs, info = shim.reset()
        log.event("reset_done", task_prompt=info.get("task_prompt"))
        snapshot_state(shim, log, "post_reset")
        save_cams(shim, out_dir, "post_reset", log)

        if args.nan_replay:
            log.phase = "nan_replay"
            rng = np.random.default_rng(0)
            base_pos, _ = shim.robot.get_position_orientation()
            base_xy = np.asarray(base_pos, dtype=np.float64)[0:2]
            for i in range(args.nan_replay):
                # Alternate wild world-frame targets around the robot: far, low,
                # high, behind — exactly the far-IK-solution motions suspected in
                # the measurement-b2 physics NaN (INVESTIGATION-PROMPT.md #2).
                ang = rng.uniform(-np.pi, np.pi)
                radius = rng.uniform(1.5, 4.0)
                target = np.array([
                    base_xy[0] + radius * np.cos(ang),
                    base_xy[1] + radius * np.sin(ang),
                    rng.uniform(0.02, 1.8),
                ])
                quat_xyzw = np.array([1.0, 0.0, 0.0, 0.0])  # top-down
                arm = int(i % 2)
                if i % 2 == 0:
                    # interpolated path (10 waypoints, repeat=3) — shim._move_hand
                    log.event("nan_replay_move", i=i, mode="move_hand", target=target)
                    ok = shim._move_hand((target, quat_xyzw), arm=arm)
                else:
                    # uninterpolated single-shot jump to the raw IK solution —
                    # what R1ProControlApi.grasp_object's move_to_joint_positions does
                    from capx.submission.shims.r1pro import _pose_world_to_base, _to_numpy, _xyzw_to_wxyz
                    b_pos, b_quat = shim.robot.get_position_orientation()
                    t_base, q_base = _pose_world_to_base(
                        _to_numpy(b_pos), _to_numpy(b_quat), target, quat_xyzw)
                    sol = shim._kinematics.solve_ik(t_base, _xyzw_to_wxyz(q_base), arm, shim._q28_now())
                    log.event("nan_replay_move", i=i, mode="single_shot", target=target,
                              max_dq=float(np.max(np.abs(sol[6:24] - shim._q28_now()[6:24]))))
                    ok = shim._move_to_joint_positions(sol, max_steps=200, settle_steps=5)
                q = shim._q28_now()
                log.event("nan_replay_result", i=i, converged=bool(ok),
                          nan=bool(np.any(np.isnan(q))))
                if np.any(np.isnan(q)):
                    log.event("nan_reproduced", i=i)
                    break
            log.event("nan_replay_done")
            client.close()
            return

        api = R1ProControlApi(shim, use_sam3=True)
        log.event("api_ready")

        # -- optional scripted route (obstacle-avoiding waypoint legs) ----
        if args.route:
            log.phase = "route"
            for leg_idx, leg in enumerate(s for s in args.route.split(";") if s.strip()):
                x_str, _, y_str = leg.partition(",")
                leg_x, leg_y = float(x_str), float(y_str)
                q = shim._q28_now()
                leg_yaw = float(np.arctan2(leg_y - q[1], leg_x - q[0]))
                t0 = time.monotonic()
                reached = shim._navigate_to_pose((leg_x, leg_y, leg_yaw))
                log.event("route_leg", idx=leg_idx, target=[leg_x, leg_y, leg_yaw],
                          reached=bool(reached), dt=round(time.monotonic() - t0, 1),
                          base=shim._q28_now()[0:6])
                save_cams(shim, out_dir, f"route{leg_idx}", log)

        # -- find the object (agent block 0 sequence) --------------------
        log.phase = "find"
        api.reset_torso()
        try:
            pose = api.get_object_pose(args.object)
        except ValueError:
            pose = (None,) * 5
        if pose[0] is None:
            try:
                api.find_object_base_rotate(args.object)
                api.find_object_torso_rotate(args.object)
            except ValueError:
                pass
            pose = api.get_object_pose(args.object)
        if pose[0] is None:
            log.event("object_not_found_abort")
            return
        obj_pos, obj_quat_wxyz, _, obj_points, obb = pose
        cam_T = api.get_camera_pose()
        log.event("object_found", obj_pos=obj_pos, obj_quat_wxyz=obj_quat_wxyz,
                  T_world_cam=np.asarray(cam_T),
                  n_points=int(np.asarray(obj_points).shape[0]))
        save_cams(shim, out_dir, "found", log)

        # -- approach until the freshest detection is reachable ----------
        if args.approach_until_reachable:
            log.phase = "approach"
            for hop in range(args.max_hops):
                robot_pos, _, robot_yaw = api.get_robot_position()
                dx, dy = obj_pos[0] - robot_pos[0], obj_pos[1] - robot_pos[1]
                dist = float(np.hypot(dx, dy))
                log.event("hop_status", hop=hop, obj_pos=obj_pos,
                          robot_pos=np.asarray(robot_pos), dist=dist)
                if dist <= args.reach_radius:
                    log.event("reachable", hop=hop, dist=dist)
                    break
                standoff = 0.35
                target = (obj_pos[0] - dx / dist * standoff, obj_pos[1] - dy / dist * standoff,
                          float(np.arctan2(dy, dx)))
                reached = api.navigate_to_pose(target)
                log.event("hop_nav_done", hop=hop, target=list(target), reached=bool(reached))
                try:
                    pose = api.get_object_pose(args.object)
                except ValueError:
                    pose = (None,) * 5
                if pose[0] is None:
                    try:
                        api.find_object_torso_rotate(args.object)
                        pose = api.get_object_pose(args.object)
                    except ValueError:
                        pose = (None,) * 5
                if pose[0] is None:
                    log.event("object_lost_in_hop", hop=hop)
                    return
                obj_pos = pose[0]
                save_cams(shim, out_dir, f"hop{hop}", log)
            else:
                log.event("never_reachable_abort")
                return

        # -- navigate to 0.35 m standoff (agent block 0) ------------------
        elif not args.skip_nav:
            log.phase = "navigate"
            robot_pos, robot_quat, robot_yaw = api.get_robot_position()
            dx, dy = obj_pos[0] - robot_pos[0], obj_pos[1] - robot_pos[1]
            dist = float(np.hypot(dx, dy))
            log.event("nav_plan", robot_pos=np.asarray(robot_pos), robot_yaw=float(robot_yaw), dist=dist)
            if dist > 0.35:
                target = (obj_pos[0] - dx / dist * 0.35, obj_pos[1] - dy / dist * 0.35, float(np.arctan2(dy, dx)))
                t0 = time.monotonic()
                reached = api.navigate_to_pose(target)
                log.event("nav_done", target=list(target), reached=bool(reached),
                          dt=round(time.monotonic() - t0, 1))
            pose = api.get_object_pose(args.object)
            if pose[0] is None:
                api.find_object_torso_rotate(args.object)
                pose = api.get_object_pose(args.object)
            if pose[0] is None:
                log.event("object_lost_after_nav_abort")
                return
            obj_pos = pose[0]
            log.event("object_after_nav", obj_pos=obj_pos)
            save_cams(shim, out_dir, "after_nav", log)

        # -- sample grasp poses ------------------------------------------
        log.phase = "sample"
        t0 = time.monotonic()
        pregrasps, grasps = api.sample_grasp_pose(args.object)
        log.event("grasp_candidates", dt=round(time.monotonic() - t0, 1),
                  n=0 if pregrasps is None else len(pregrasps),
                  pregrasps=[[p[0], p[1]] for p in (pregrasps or [])],
                  grasps=[[g[0], g[1]] for g in (grasps or [])])
        if not pregrasps:
            log.event("no_grasp_candidates_abort")
            return

        # -- execute (candidate order = agent order) ----------------------
        for i, (pre, gr) in enumerate(zip(pregrasps, grasps)):
            if i >= args.max_attempts:
                break
            ok = instrumented_grasp(api, shim, log, out_dir, pre, gr, tag=f"cand{i}")
            log.event("attempt_result", idx=i, in_hand=bool(ok))
            if ok:
                break
            api.open_gripper(arm=0)

        log.phase = "done"
        log.event("probe_done")
        client.close()


if __name__ == "__main__":
    main()
