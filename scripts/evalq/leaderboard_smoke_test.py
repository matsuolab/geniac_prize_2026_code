"""End-to-end smoke test of the queue and the public leaderboard, with three HF test repos.

    leaderboard_smoke_test.py setup [--profile beta] --base TEST_CONFIG --deployed CONFIG [--out DIR]
    leaderboard_smoke_test.py run   [--profile beta] --config SMOKE_CONFIG --images 1:I,H 2:K,E 3:F
    leaderboard_smoke_test.py check --config SMOKE_CONFIG

Two profiles run the same scenario:

    alpha  three real runs on small packs (50 min of benchmark clock each): ~2 h 40 min
    beta   one real run on the deployed benchmarks (10 h 35 min): ~11 h. The other runs
           that reach a worker crash on purpose, so every status is still seen.

``setup`` copies the deployed queue config (``--deployed``) and takes only ``root``,
``teams`` (the registry) and ``leaderboard.repo`` from the test config (``--base``); the publisher
heartbeats every 5 min. alpha swaps in packs cut from the deployed ones (the first
tasks of each subset, 2 trials each, 5 min per trial; see PACKS), scored as
``robosuite-2x2``, ``libero-2x2`` and ``behavior-1x2`` so a smoke board is never
mistaken for the real one, and writes ``DIR/smoke.yaml`` (``~/capx-queue/smoke``).
beta writes ``DIR/beta.yaml`` (``~/capx-queue/beta``). Start the jobs on that config:

    CAPX_EVALQ_PYTHON=... CAPX_QT_CONFIG=DIR/smoke.yaml sbatch -p part-$(id -gn) \\
        --job-name=auto-eval-pipeline-test scripts/evalq/queue-test.sbatch
    CAPX_EVALQ_PYTHON=... CAPX_QT_CONFIG=DIR/smoke.yaml sbatch -p part-$(id -gn) \\
        --job-name=auto-eval-leaderboard-test scripts/evalq/leaderboard-publisher.sbatch

``run`` needs a queue that has not seen the images it pushes (a fresh root, or the old
``results/``, ``queue/`` and ``state/`` moved to ``archive/``; never deleted), with no
watcher running yet: start ``run`` first and the two jobs once it has pushed step 1, or
the watcher evaluates whatever the repos pointed at before. ``--images``
names, per team, image files already in that team's repo (``agent-<NAME>.sif``): the
first two of team 1 and team 2 and the first of team 3 are used. It pushes, then waits
on the published ``leaderboard.json`` for each expected state:

    1. a push per team          -> running, waiting #1, waiting #2
    2. while team 1 runs:
       team 1 pushes again      -> deferred (queued when its run ends)
       team 2 pushes a crashing -> its waiting one cancelled, the new one waiting,
              submission           later failed
       team 3 pushes 3 manifests -> rejected x3, its waiting one untouched (4 rows:
              with missing images     "Show more")
    3. every run finishes       -> done / over-time, scores, ranking, one best tag per
                                   ranked team (``done`` is only a warning if never seen:
                                   a run that hits its clock is over-time)

beta crashes team 3's first push and team 1's deferred one (they end ``failed``).

``check`` asserts what is true at any point: one status per row, one best tag per ranked
team on its highest score, nothing private in the document, and the subsets and time
limit the page's subtitle is built from. It exits non-zero on a failure. Results stay on
the cluster under the queue root.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import shlex
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from push_test_submission import ENV, WEIGHTS

REPO_ROOT = Path(__file__).resolve().parents[2]
SPACE_URL = "https://{owner}-{name}.static.hf.space/leaderboard.json"
STATUSES = ("done", "running", "waiting", "deferred", "cancelled", "over-time", "rejected", "failed")
# Field names that would leak the pack, the budgets or the cluster if they were ever published.
# Not error_code: it is an enum, and the ranking view publishes it on purpose.
FORBIDDEN_KEYS = {"tasks", "success_rate", "subsets", "episode_timeout_s", "job_timeout_s", "subset_timeout_s",
                  "error", "image", "image_path", "hub_repo", "hub_revision", "logs_dir"}

# Two tasks per benchmark (one for behavior), two trials each, every trial of a benchmark at
# once (one lane per trial). The pack's budget is 5 min per trial (``limits.trial_time_s``),
# so 20 + 20 + 10 min. The benchmarks still run one after another.
TRIAL_S = 300.0
# Tasks per deployed subset, the first ones of its pack.
PACKS = {
    "robosuite": 2,
    "libero": 2,
    # One BEHAVIOR task: a second activity crashes the env process that already loaded the
    # first scene ("Simulator must be stopped before loading scene!", r1pro_b1k.py).
    "behavior": 1,
}
SUBSETS = [f"{name}-{tasks}x2" for name, tasks in PACKS.items()]
TIME_LIMIT_S = TRIAL_S * 2 * sum(PACKS.values())
CRASH = "sh -c 'echo smoke test: failing on purpose; exit 3'"


# --- setup -----------------------------------------------------------------------------

def _test_config(args: argparse.Namespace) -> dict:
    """The deployed config, pointed at the test teams, root and Space of ``--base``."""
    base = yaml.safe_load(Path(args.base).expanduser().read_text())
    config = yaml.safe_load(Path(args.deployed).expanduser().read_text())
    config["root"] = base["root"]
    config["teams"] = copy.deepcopy(base["teams"])
    config.setdefault("leaderboard", {}).update(repo=base["leaderboard"]["repo"], heartbeat_s=300.0, interval_s=60.0)
    return config


def setup(args: argparse.Namespace) -> None:
    config = _test_config(args)
    if args.profile == "beta":
        out = Path(args.out or "~/capx-queue/beta").expanduser()
        out.mkdir(parents=True, exist_ok=True)
        target = out / "beta.yaml"
        target.write_text(f"# Leaderboard smoke test, beta: {args.deployed} on the test teams of {args.base}\n"
                          "# (scripts/evalq/leaderboard_smoke_test.py setup --profile beta).\n"
                          + yaml.safe_dump(config, sort_keys=False))
        print(f"wrote {target}: {', '.join(s['name'] for s in config['subsets'])}, "
              f"time limit {_duration(_time_limit_s(str(target)))}")
        return
    out = Path(args.out or "~/capx-queue/smoke").expanduser()
    subsets = []
    for deployed in config["subsets"]:
        count = PACKS.get(deployed["name"])
        if not count:
            continue
        pack_path = Path(deployed["taskpack"]).expanduser()
        if not pack_path.is_absolute():
            pack_path = REPO_ROOT / pack_path
        source = yaml.safe_load((pack_path / "pack.yaml").read_text())
        simulators = deployed.get("simulators") or []
        tasks = [t for t in source["tasks"] if not simulators or t["simulator"] in simulators][:count]
        subset = f"{deployed['name']}-{count}x2"
        pack_dir = out / f"pack-{count}x2-{deployed['name']}"
        pack_dir.mkdir(parents=True, exist_ok=True)
        pack = {"pack_id": f"smoke-{subset}", "version": "1", "visibility": source.get("visibility", "public"),
                "limits": {**source["limits"], "trial_time_s": TRIAL_S, "n_trials": 2},
                "tasks": [{**copy.deepcopy(t), "n_trials": 2, "seed_policy": {"mode": "fixed", "seeds": [0, 1]}}
                          for t in tasks]}
        (pack_dir / "pack.yaml").write_text(
            f"# Smoke test: {count} task(s) x 2 trials, {TRIAL_S:.0f} s per trial, from {pack_path}.\n"
            + yaml.safe_dump(pack, sort_keys=False))
        subsets.append({"name": subset, "taskpack": str(pack_dir), "simulators": simulators})
        for simulator in simulators:
            if simulator in config.get("simulators", {}):
                config["simulators"][simulator]["max_lanes"] = count * 2
    if [s["name"] for s in subsets] != [f"{n}-{c}x2" for n, c in PACKS.items()]:
        raise SystemExit(f"{args.deployed} does not have the subsets {', '.join(PACKS)}")

    config["subsets"] = subsets
    config["leaderboard"].pop("subset_labels", None)  # the -NxM names are what mark a smoke board
    target = out / "smoke.yaml"
    target.write_text("# Leaderboard smoke test (scripts/evalq/leaderboard_smoke_test.py setup).\n"
                      + yaml.safe_dump(config, sort_keys=False))
    print(f"wrote {target} ({_duration(_time_limit_s(str(target)))}) and "
          f"{', '.join(s['taskpack'] for s in subsets)}")


def _time_limit_s(config_path: str) -> float:
    """The total job budget the page publishes, from the packs, as the queue computes it."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from capx.evalqueue.config import load_queue_config
    from capx.protocol.taskpack import load_pack

    queue = load_queue_config(str(Path(config_path).expanduser()))
    return sum(load_pack(queue.resolve(s.taskpack))[0].job_timeout_s(s.simulators or None)
               for s in queue.subsets)


def expected_scoring(config_path: str) -> tuple[list[str], float]:
    """The subsets (under their labels) and total time limit this config should publish,
    which the page's subtitle is built from."""
    config = _config(config_path)
    labels = config.get("leaderboard", {}).get("subset_labels") or {}
    return [labels.get(s["name"], s["name"]) for s in config["subsets"]], _time_limit_s(config_path)


# --- reading the board -------------------------------------------------------------------

def _config(path: str) -> dict:
    return yaml.safe_load(Path(path).expanduser().read_text())


def _url(config: dict) -> str:
    owner, name = config["leaderboard"]["repo"].split("/")
    return SPACE_URL.format(owner=owner, name=name)


def fetch(config: dict) -> dict:
    with urllib.request.urlopen(f"{_url(config)}?t={time.time()}", timeout=30) as response:
        return json.load(response)


def rows(board: dict) -> dict[str, list[dict]]:
    return {team["team_id"]: team["activity"] for team in board["queue"]["teams"]}


def scoring(board: dict) -> str:
    """What the page's subtitle is built from (web/leaderboard/index_en.html, show())."""
    limit = board["scoring"].get("time_limit_s")
    return f"{', '.join(board['scoring']['subsets'])} · limit {_duration(limit) if limit else 'none'}"


def _duration(seconds: float) -> str:
    """The page's dur(): "45 min", "1 h 20 min"."""
    minutes = round(seconds / 60)
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60:02d} min"


def problems(board: dict, subsets: list[str] = SUBSETS, time_limit_s: float | None = TIME_LIMIT_S) -> list[str]:
    found = []
    if board["scoring"]["subsets"] != subsets:
        found.append(f"the page names the subsets {board['scoring']['subsets']}, not {subsets}")
    if time_limit_s is not None and board["scoring"].get("time_limit_s") != time_limit_s:
        found.append(f"the time limit is {board['scoring'].get('time_limit_s')} s, not {time_limit_s:.0f} s")
    keys: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            keys.update(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk({"leaderboard": board["leaderboard"], "queue": board["queue"]})
    found += [f"published document has a {key!r} field" for key in sorted(keys & FORBIDDEN_KEYS)]
    if "/home/" in json.dumps(board):
        found.append("published document contains a /home/ path")
    for team, activity in rows(board).items():
        for row in activity:
            if row["status"] not in STATUSES:
                found.append(f"{team}: unknown status {row['status']!r}")
        ranked = [row for row in activity if row.get("ranked")]
        entry = next(e for e in board["leaderboard"] if e["team_id"] == team)
        if entry["rank"] is not None and [r["submission_id"] for r in ranked] != [entry["submission_id"]]:
            found.append(f"{team}: best tag on {[r['submission_id'] for r in ranked]}, ranked by {entry['submission_id']}")
        if entry["rank"] is None and ranked:
            found.append(f"{team}: best tag without a rank")
    ranks = [e["rank"] for e in board["leaderboard"] if e["rank"] is not None]
    if ranks != list(range(1, len(ranks) + 1)):
        found.append(f"ranks are {ranks}")
    return found


def show(board: dict) -> None:
    print(f"generated {board['generated_at']}  scoring: {scoring(board)}")
    for e in board["leaderboard"]:
        score = "-" if e["score"] is None else f"{e['score'] * 100:.1f}%"
        print(f"  #{e['rank'] or '-'} {e['team_id']:<12} {e.get('submission_id') or '':<20} {score:>6}"
              f"  {(e.get('time_taken_s') or 0) / 60:5.1f} min")
    for team, activity in rows(board).items():
        print(f"  {team}: " + ", ".join(
            f"{r['submission_id']}={r['status']}{'#' + str(r['position']) if r.get('position') else ''}"
            f"{' [best]' if r.get('ranked') else ''}" for r in activity))


def check(args: argparse.Namespace) -> None:
    board = fetch(_config(args.config))
    show(board)
    found = problems(board, args.expect_subsets.split(","), args.expect_time_limit_s)
    for line in found:
        print("FAIL", line)
    print("OK" if not found else f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)


# --- run ----------------------------------------------------------------------------------

class Scenario:
    def __init__(self, config_path: str, images: dict[str, list[str]], tag: str) -> None:
        from huggingface_hub import HfApi

        from capx.evalqueue.config import load_queue_config
        from capx.evalqueue.store import EvalStore
        from capx.evalqueue.teams import TeamRegistry

        self.config = _config(config_path)
        self.config_path = config_path
        queue_config = load_queue_config(config_path)
        registry = TeamRegistry(queue_config.teams, EvalStore(queue_config.root))
        self.teams = {team.team_id: team.repo for team in registry.refresh()}
        self.team_ids = list(self.teams)
        self.images = {self.team_ids[int(n) - 1]: names for n, names in images.items()}
        self.tag = tag
        self.api = HfApi()
        self.seen: dict[tuple[str, str], set[str]] = {}
        self.failures: list[str] = []

    def push(self, team: str, submission_id: str, image: str, entrypoint: str | None = None) -> None:
        repo = self.teams[team]
        manifest = {"team_id": team, "submission_id": submission_id,
                    "image": {"repo": repo, "revision": self.api.list_repo_commits(repo)[0].commit_id,
                              "path": f"agent-{image}.sif"},
                    "notes": f"leaderboard smoke test {self.tag}"}
        if image in self.images.get(team, []):  # a real image runs the reference agent
            manifest.update(weights=WEIGHTS, env=ENV)
        if entrypoint:
            manifest["entrypoint"] = shlex.split(entrypoint)
        self.api.upload_file(path_or_fileobj=io.BytesIO(yaml.safe_dump(manifest, sort_keys=False).encode()),
                             path_in_repo="submission.yaml", repo_id=repo, commit_message=submission_id)
        print(f"  pushed {team} {submission_id} (agent-{image}.sif{', crashing' if entrypoint else ''})")

    def sid(self, name: str) -> str:
        return f"{self.tag}-{name}"

    def wait(self, what: str, want: dict[str, str], timeout_s: float) -> dict:
        """Poll the board until every submission in ``want`` has one of its statuses
        ("done/over-time" accepts either), recording every status seen on the way."""
        deadline = time.time() + timeout_s
        print(f"- waiting for {what} (up to {timeout_s / 60:.0f} min)")
        while True:
            board = fetch(self.config)
            status = {r["submission_id"]: r["status"] for acts in rows(board).values() for r in acts}
            for team, activity in rows(board).items():
                for r in activity:
                    self.seen.setdefault((team, r["submission_id"]), set()).add(r["status"])
            if all(status.get(s) in wanted.split("/") for s, wanted in want.items()):
                print(f"  ok: {what}")
                return board
            if time.time() > deadline:
                missing = {s: (status.get(s), w) for s, w in want.items() if status.get(s) not in w.split("/")}
                self.fail(f"{what}: timed out, (actual, wanted) = {missing}")
                return board
            time.sleep(30)

    def fail(self, message: str) -> None:
        print("  FAIL", message)
        self.failures.append(message)

    def run(self, profile: str) -> None:
        t1, t2, t3 = self.team_ids
        a1, b1 = self.images[t1][:2]
        a2, b2 = self.images[t2][:2]
        a3 = self.images[t3][0]
        print("step 1: one push per team")
        self.push(t1, self.sid("t1-first"), a1)
        self.push(t2, self.sid("t2-first"), a2)
        beta = profile == "beta"  # one real run: what else reaches a worker crashes
        self.push(t3, self.sid("t3-first"), a3, entrypoint=CRASH if beta else None)
        self.wait("running / waiting / waiting",
                  {self.sid("t1-first"): "running", self.sid("t2-first"): "waiting", self.sid("t3-first"): "waiting"},
                  45 * 60)

        print("step 2: pushes while team 1 runs")
        self.push(t1, self.sid("t1-deferred"), b1, entrypoint=CRASH if beta else None)
        self.push(t2, self.sid("t2-crash"), b2, entrypoint=CRASH)
        for n in (1, 2, 3):
            self.push(t3, self.sid(f"t3-missing-{n}"), f"missing-{n}")
            time.sleep(75)  # one push per watcher poll, so each is seen and rejected
        self.wait("deferred, cancelled + waiting, rejected x3",
                  {self.sid("t1-deferred"): "deferred", self.sid("t2-first"): "cancelled",
                   self.sid("t2-crash"): "waiting/running", self.sid("t3-first"): "waiting/running",
                   **{self.sid(f"t3-missing-{n}"): "rejected" for n in (1, 2, 3)}}, 20 * 60)

        print("step 3: every run finishes")
        # The real runs go one after another, each its benchmarks' clock plus start-up.
        limit = float(fetch(self.config)["scoring"].get("time_limit_s") or 0)
        real = "failed" if beta else "done/over-time"
        board = self.wait("all runs finished", {
            self.sid("t1-first"): "done/over-time", self.sid("t1-deferred"): real,
            self.sid("t2-crash"): "failed", self.sid("t3-first"): real},
            max(12 * 3600, (1 if beta else 3) * (limit + 3600)))

        print("step 4: what the page shows")
        show(board)
        for line in problems(board, *expected_scoring(self.config_path)):
            self.fail(line)
        everything = set().union(*self.seen.values())
        for status in STATUSES:
            if status in everything:
                print(f"  status seen: {status}")
            elif status == "done":
                print("  WARN status never seen: done (every real run hit its clock)")
            else:
                self.fail(f"status never seen: {status}")
        if len(rows(board)[t3]) <= int(self.config["leaderboard"].get("activity_per_team", 3)):
            self.fail("team 3 has too few rows to show the Show more button")
        print("PASS" if not self.failures else f"{len(self.failures)} FAILURE(S):\n  " + "\n  ".join(self.failures))
        sys.exit(1 if self.failures else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--profile", choices=("alpha", "beta"), default="alpha",
                   help="alpha: small packs (~2 h 40 min); beta: the deployed benchmarks (~11 h)")
    p.add_argument("--base", required=True, help="the test config: root, test teams and leaderboard Space")
    p.add_argument("--deployed", required=True, help="the deployed queue config everything else comes from")
    p.add_argument("--out", help="default ~/capx-queue/smoke (alpha) or ~/capx-queue/beta (beta)")
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--images", nargs=3, required=True, metavar="N:NAME,NAME",
                   help="per team, image names already in its repo, e.g. 1:I,H 2:K,E 3:F")
    p.add_argument("--profile", choices=("alpha", "beta"), default="alpha",
                   help="beta: one real run, the other runs crash on purpose")
    p.add_argument("--tag", default=time.strftime("st%m%d%H%M"), help="prefix of the submission ids")
    p = sub.add_parser("check")
    p.add_argument("--config", required=True)
    p.add_argument("--expect-subsets", default=",".join(SUBSETS),
                   help="the subset names the page must show, comma-separated (default: the smoke test's)")
    p.add_argument("--expect-time-limit-s", type=float, default=TIME_LIMIT_S,
                   help="the evaluation time limit the page must show, in seconds (default: the smoke test's)")
    args = parser.parse_args()
    if args.command == "setup":
        setup(args)
    elif args.command == "check":
        check(args)
    else:
        images = dict(item.split(":", 1) for item in args.images)
        Scenario(args.config, {n: names.split(",") for n, names in images.items()}, args.tag).run(args.profile)


if __name__ == "__main__":
    main()
