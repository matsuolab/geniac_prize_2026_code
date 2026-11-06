"""Generate the original task pack: what the CaP-X paper and its release evaluate.

The paper's core analysis runs 7 Robosuite tasks, and the full CaP-Gym release ships
187 (7 Robosuite + 130 LIBERO-PRO + 50 BEHAVIOR). This writes the part of that which
the gateway can serve:

- **All 7 Robosuite tasks**, the five single-arm ones plus Two-Arm Lift and Two-Arm
  Handover over the ``franka_bimanual`` embodiment.
- **The 130 LIBERO-PRO tasks**, enumerated from the benchmark itself so the pack
  cannot drift from what is installed. ``--libero-per-suite N`` keeps an evenly
  spaced N from each of the five suites instead, for when the trial count per task
  matters more than the task count (a rate over 130 tasks x 1 trial is noise).
- **The 50 BEHAVIOR tasks**: the 48 BEHAVIOR-1K activities of ``env_configs/r1pro/b1k_*.yaml``
  (each passed to the env as ``activity_name``), ``picking_up_trash`` and the radio pickup,
  which runs on its own controller config. The eval queue's default subset selects
  robosuite and libero only; BEHAVIOR needs an RTX-class GPU for Isaac Sim.

Robosuite tasks get ``--robosuite-trials`` trials each with distinct seeds; LIBERO and
BEHAVIOR get ``--n-trials``. The job budget follows from the pack: ``trial_time_s`` for
every trial.

Needs LIBERO importable, so it runs inside the env-libero image:

    sbatch -p part-$(id -gn) scripts/evalq/build-original-pack.sbatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# The five single-arm Robosuite tasks of the paper's core set. Instructions are the
# goal sentence from each task's own prompt (capx/envs/tasks/franka/*.py) — the pack
# states the goal; how to pursue it is the submission's business.
ROBOSUITE_TASKS = [
    {
        "task_id": "cube_lifting",
        "env_target": "franka_robosuite_cube_lift_low_level",
        "trial_steplimit": 1500,
        "instruction": "Pick up the red cube and lift it. Quaternions are WXYZ.",
    },
    {
        "task_id": "cube_stack",
        "env_target": "franka_robosuite_cubes_low_level",
        "trial_steplimit": 1500,
        "instruction": (
            "Pick up the red cube and gently stack it on top of the green cube, then "
            "release it. Quaternions are WXYZ."
        ),
    },
    {
        "task_id": "cube_restack",
        "env_target": "franka_robosuite_cubes_restack_low_level",
        "trial_steplimit": 1500,
        "instruction": (
            "Gently place the red cube on top of the green cube and then open the "
            "gripper. Nothing should be dropped from a height. Quaternions are WXYZ."
        ),
    },
    {
        "task_id": "nut_assembly",
        "env_target": "franka_robosuite_nut_assembly_low_level",
        "trial_steplimit": 1000,
        "instruction": (
            "Grasp the brown square nut by its handle and insert it onto the brown "
            "square block. Quaternions are WXYZ."
        ),
    },
    {
        "task_id": "spill_wipe",
        "env_target": "franka_robosuite_spill_wipe_low_level",
        "trial_steplimit": 4000,
        "instruction": (
            "Wipe up the brown spill. A sponge is already attached to the "
            "end-effector, and the table surface is at z = 0.0 m. Quaternions are WXYZ."
        ),
    },
]

# The other two of the paper's seven. Two arms, so a different embodiment, action
# and camera: these render a scene camera ("agentview") and no wrist view.
BIMANUAL_TASKS = [
    {
        "task_id": "two_arm_lift",
        "env_target": "two_arm_lift_robosuite",
        "trial_steplimit": 5000,
        "instruction": (
            "The two arms must coordinate to lift a pot. Arm 0 grasps handle 0 and "
            "arm 1 grasps handle 1, then both lift the pot. Quaternions are WXYZ."
        ),
    },
    {
        "task_id": "two_arm_handover",
        "env_target": "two_arm_handover_robosuite",
        "trial_steplimit": 5000,
        "instruction": (
            "Arm 0 picks up the hammer, lifts it, and hands it over to arm 1, which "
            "grasps it by the handle rather than the head. Quaternions are WXYZ."
        ),
    },
]

BIMANUAL_CAMERAS = ["agentview"]

ROBOSUITE_CAMERAS = ["robot0_robotview"]
# LIBERO's scene camera plus the wrist camera; also what the submission runner keys
# on to pick the LIBERO task config (an "agentview" camera means LIBERO).
LIBERO_CAMERAS = ["agentview", "robot0_eye_in_hand"]

BEHAVIOR_TASKS = [
    {
        "task_id": "picking_up_trash",
        "instruction": "Find the can of soda on the floor and pick it up.",
        "env_kwargs": {},
    },
    {
        "task_id": "picking_up_radio",
        "instruction": "Pick up the red radio on the table.",
        "env_kwargs": {"controller_cfg": "r1pro_primitives_radio.yaml"},
    },
]
BEHAVIOR_CAMERAS = ["ego", "left_wrist", "right_wrist"]
# The BEHAVIOR-1K activities CaP-X ships agent configs for; each names its activity
# and states its goal in a "Goal:" line of the prompt.
B1K_CONFIGS = Path(__file__).resolve().parents[2] / "env_configs" / "r1pro"

# The 130 LIBERO-PRO tasks, as registered in capx/envs/simulators/__init__.py.
LIBERO_SUITES = ["libero_10", "libero_90", "libero_object", "libero_spatial", "libero_goal"]
# The pack's step limit: the sim horizon of CaP-X's own LIBERO agent configs
# (env_configs/libero/*cap_agent0.yaml). Robosuite tasks keep the horizons their envs
# always had.
DEFAULT_STEPLIMIT = 8000


def _spread(n_tasks: int, keep: int) -> list[int]:
    """`keep` evenly spaced indices out of `n_tasks`, endpoints included.

    Evenly spaced rather than the first N or a random N: the suites are ordered by
    scene, so the first N would all share a kitchen, and a random N would not be the
    same sample twice.
    """
    if keep <= 0 or keep >= n_tasks:
        return list(range(n_tasks))
    if keep == 1:
        return [0]
    return sorted({round(i * (n_tasks - 1) / (keep - 1)) for i in range(keep)})


def libero_tasks(n_trials: int, per_suite: int = 0) -> list[dict]:
    from libero import benchmark  # type: ignore[import-not-found]

    benchmark_dict = benchmark.get_benchmark_dict()
    tasks = []
    for suite_name in LIBERO_SUITES:
        suite = benchmark_dict[suite_name]()
        for index in _spread(suite.n_tasks, per_suite):
            task = suite.get_task(index)
            language = (task.language or "").strip().rstrip(".")
            # A LIBERO task does not randomize its scene: it replays one of a fixed
            # list of recorded initial states, and FrankaLiberoEnv wraps around that
            # list (state_idx = (seed - 1) % len(init_states)). Asking for more
            # trials than there are states re-runs states already seen, which buys
            # sampling noise rather than another initial condition.
            states = suite.get_task_init_states(index)
            trials = min(n_trials, len(states)) if states is not None else n_trials
            fewer = {"n_trials": trials} if trials != n_trials else {}
            tasks.append(
                {
                    "task_id": f"{suite_name}_{index}",
                    "embodiment": "franka",
                    "simulator": "libero",
                    "instruction": f"{language.capitalize()}. Quaternions are WXYZ.",
                    "camera_names": list(LIBERO_CAMERAS),
                    "env_target": f"franka_libero_{suite_name}_{index}_low_level",
                    "privileged": False,
                    **fewer,
                    # FrankaLiberoEnv.reset() treats the seed as a 1-based init-state
                    # index, so seed 0 would wrap to the LAST init state instead of
                    # the first.
                    "seed_policy": {
                        "mode": "fixed",
                        "seeds": list(range(1, trials + 1)),
                    },
                }
            )
    return tasks


def robosuite_tasks(n_trials: int) -> list[dict]:
    single = [
        (task, "franka", ROBOSUITE_CAMERAS) for task in ROBOSUITE_TASKS
    ]
    both = single + [
        (task, "franka_bimanual", BIMANUAL_CAMERAS) for task in BIMANUAL_TASKS
    ]
    return [
        {
            "task_id": task["task_id"],
            "embodiment": embodiment,
            "simulator": "robosuite",
            "instruction": task["instruction"],
            "camera_names": list(cameras),
            "env_target": task["env_target"],
            "privileged": False,
            "n_trials": n_trials,
            "trial_steplimit": task["trial_steplimit"],
            "seed_policy": {"mode": "fixed", "seeds": list(range(n_trials))},
        }
        for task, embodiment, cameras in both
    ]


def b1k_tasks() -> list[dict]:
    tasks = []
    for path in sorted(B1K_CONFIGS.glob("b1k_*.yaml")):
        cfg = yaml.safe_load(path.read_text())["env"]["cfg"]
        activity = cfg["low_level"]["activity_name"]
        goal = next(
            line.strip().removeprefix("Goal:").strip()
            for line in cfg["prompt"].splitlines()
            if line.strip().startswith("Goal:")
        )
        tasks.append(
            {"task_id": activity, "instruction": goal, "env_kwargs": {"activity_name": activity}}
        )
    return tasks


def behavior_tasks(n_trials: int) -> list[dict]:
    # A BEHAVIOR seed is a presampled task-instance index, so it must be fixed.
    return [
        {
            "task_id": task["task_id"],
            "embodiment": "r1pro",
            "simulator": "behavior",
            "instruction": task["instruction"],
            "camera_names": list(BEHAVIOR_CAMERAS),
            "env_target": "r1pro_b1k_low_level",
            **({"env_kwargs": task["env_kwargs"]} if task["env_kwargs"] else {}),
            "privileged": False,
            "seed_policy": {"mode": "fixed", "seeds": list(range(n_trials))},
        }
        for task in BEHAVIOR_TASKS + b1k_tasks()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="taskpacks/original/pack.yaml")
    parser.add_argument("--n-trials", type=int, default=1, help="LIBERO and BEHAVIOR")
    parser.add_argument("--robosuite-trials", type=int, default=10)
    parser.add_argument("--trial-time-s", type=float, default=300.0)
    parser.add_argument(
        "--skip-libero",
        action="store_true",
        help="Leave out the LIBERO tasks (for checking the script without LIBERO installed).",
    )
    parser.add_argument(
        "--libero-per-suite",
        type=int,
        default=0,
        help="Keep this many evenly spaced tasks from each LIBERO suite (0 = all 130).",
    )
    parser.add_argument("--pack-id", default="original")
    args = parser.parse_args()

    tasks = robosuite_tasks(args.robosuite_trials)
    if not args.skip_libero:
        tasks += libero_tasks(args.n_trials, args.libero_per_suite)
    tasks += behavior_tasks(args.n_trials)

    pack = {
        "pack_id": args.pack_id,
        "version": "0.2",
        "visibility": "public",
        "limits": {
            "trial_time_s": args.trial_time_s,
            "n_trials": args.n_trials,
            "trial_steplimit": DEFAULT_STEPLIMIT,
        },
        "tasks": tasks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(pack, sort_keys=False, width=100, allow_unicode=True))
    by_sim: dict[str, int] = {}
    for task in tasks:
        by_sim[task["simulator"]] = by_sim.get(task["simulator"], 0) + 1
    print(f"wrote {out} with {len(tasks)} task(s): {by_sim}", file=sys.stderr)


if __name__ == "__main__":
    main()
