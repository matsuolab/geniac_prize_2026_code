"""Stress the intake: many submissions from a few teams in a short time.

    leaderboard_stress_test.py setup  --total 1000 [--teams 3] [--tag TAG]
    leaderboard_stress_test.py run    --total 1000 --minutes 60 [--teams 3] [--tag TAG]
    leaderboard_stress_test.py report --config QUEUE.yaml [--tag TAG]

``setup`` commits, per team repo, one tiny unique image per submission
(``stress-TAG/NNNN.sif``) in a single commit. They are too small for LFS, so the queue
keys them by manifest commit rather than image hash; either way each push is distinct. ``run`` then pushes one ``submission.yaml`` per submission,
spread evenly over ``--minutes`` with one thread per team, logging each push to
``~/capx-queue/stress-TAG/pushes.jsonl``. The images are a few KB of random bytes, so
intake costs no real download and every evaluation that starts fails at the submission
step; what is under test is the watcher, the queue and the board, not the agent.

``report`` joins the push log with the queue store: how many pushes the watcher saw,
what became of each (enqueued, cancelled, ran), the per-team cap, intake latency,
and the published board's size.

The Hub refuses a repo's 129th commit in an hour, so N teams can land at most ~128·N
pushes an hour (setup's commit counts); a team that hits it stops.

Settings as push_test_submission.py: CAPX_QT_OWNER, CAPX_QT_REPO_PREFIX,
CAPX_QT_TEAM_PREFIX.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError

from push_test_submission import OWNER, REPO_PREFIX, TEAM_PREFIX, WEIGHTS


def _repo(n: int) -> str:
    return f"{OWNER}/{REPO_PREFIX}-{n}"


def _workdir(tag: str) -> Path:
    path = Path(f"~/capx-queue/stress-{tag}").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plan(total: int, teams: int) -> dict[int, list[int]]:
    """Submission numbers per team, dealt round robin: 1 → team 1, 2 → team 2, ..."""
    return {n: list(range(n, total + 1, teams)) for n in range(1, teams + 1)}


def setup(args: argparse.Namespace) -> None:
    api = HfApi()
    revisions = {}
    for n, numbers in _plan(args.total, args.teams).items():
        repo = _repo(n)
        ops = [CommitOperationAdd(f"stress-{args.tag}/{i:04d}.sif", os.urandom(4096)) for i in numbers]
        info = api.create_commit(repo, ops, commit_message=f"stress {args.tag}: {len(numbers)} images")
        revisions[n] = info.oid
        print(f"{repo}: {len(numbers)} images @ {info.oid[:8]}")
    (_workdir(args.tag) / "images.json").write_text(json.dumps(revisions))


def run(args: argparse.Namespace) -> None:
    work = _workdir(args.tag)
    revisions = {int(k): v for k, v in json.loads((work / "images.json").read_text()).items()}
    interval = args.minutes * 60 / args.total
    start = time.time() + 5
    log = (work / "pushes.jsonl").open("a")
    lock = threading.Lock()
    api = HfApi()

    def team(n: int, numbers: list[int]) -> None:
        repo = _repo(n)
        for i in numbers:
            planned = start + (i - 1) * interval
            time.sleep(max(0.0, planned - time.time()))
            submission_id = f"stress-{args.tag}-{i:04d}"
            manifest = {
                "team_id": f"{TEAM_PREFIX}-{n}",
                "submission_id": submission_id,
                "image": {"repo": repo, "revision": revisions[n], "path": f"stress-{args.tag}/{i:04d}.sif"},
                "notes": f"stress test {args.tag}",
                "weights": WEIGHTS,
            }
            body = yaml.safe_dump(manifest, sort_keys=False).encode()
            row = {"i": i, "team": n, "submission_id": submission_id, "planned": planned, "retries": 0}
            t0 = time.time()
            while True:
                try:
                    info = api.upload_file(path_or_fileobj=io.BytesIO(body), path_in_repo="submission.yaml",
                                           repo_id=repo, commit_message=submission_id)
                    row.update(ok=True, commit=info.oid)
                    break
                except HfHubHTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status in (500, 502, 503, 504) and row["retries"] < 3:
                        row["retries"] += 1
                        time.sleep(2 ** row["retries"])
                        continue
                    row.update(ok=False, error=f"{status}: {str(exc)[:200]}")
                    break
            row.update(pushed=time.time(), latency=round(time.time() - t0, 2))
            with lock:
                log.write(json.dumps(row) + "\n")
                log.flush()
                if i % 30 == 0 or not row["ok"]:
                    late = row["pushed"] - planned
                    print(f"{datetime.now():%H:%M:%S} #{i} team {n} ok={row['ok']} "
                          f"latency {row['latency']}s late {late:.0f}s", flush=True)
            if "429" in row.get("error", ""):
                # The Hub's 128 commits per repo per hour: nothing more lands for an hour.
                print(f"team {n}: hourly commit cap reached after #{i}; stopping it", flush=True)
                return

    threads = [threading.Thread(target=team, args=item) for item in _plan(args.total, args.teams).items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("all pushes done", flush=True)


def _ts(value: str | None) -> float | None:
    return datetime.fromisoformat(value).timestamp() if value else None


def report(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).expanduser().read_text())
    root = Path(config["root"]).expanduser()
    prefix = f"stress-{args.tag}-"
    pushes = [json.loads(line) for line in (_workdir(args.tag) / "pushes.jsonl").read_text().splitlines()]
    pushed = {p["submission_id"]: p for p in pushes if p["ok"]}

    results = {}
    for path in (root / "results").glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("submission_id", "")).startswith(prefix):
            results[data["submission_id"]] = data
    jobs = {}
    for state in ("pending", "claimed", "done", "failed"):
        for path in (root / "queue" / state).glob("*.json"):
            data = json.loads(path.read_text())
            if str(data.get("submission_id", "")).startswith(prefix):
                jobs[data["submission_id"]] = (state, data)

    seen = set(results) | set(jobs)
    statuses = Counter(r["status"] for r in results.values())
    span = max(p["pushed"] for p in pushes) - min(p["pushed"] for p in pushes)
    latencies = [p["latency"] for p in pushes if p["ok"]]
    print(f"pushes: {len(pushes)} ({len(pushed)} ok, {len(pushes) - len(pushed)} failed, "
          f"{sum(p['retries'] for p in pushes)} retries) over {span / 60:.1f} min; "
          f"commit latency median {statistics.median(latencies):.1f}s, max {max(latencies):.1f}s")
    print(f"picked up by the watcher: {len(seen)} ({len(seen) / max(1, len(pushed)):.0%}); "
          f"the rest were overwritten by the next push before a poll saw them")
    print(f"results: {dict(statuses)}; still queued: {Counter(s for s, _ in jobs.values() if s in ('pending', 'claimed'))}")

    intake = []
    for sid, (_, job) in jobs.items():
        if sid in pushed and job.get("enqueued_at"):
            intake.append(_ts(job["enqueued_at"]) - pushed[sid]["pushed"])
    for sid, result in results.items():
        if sid in pushed and result.get("status") == "cancelled" and result.get("submitted_at"):
            intake.append(_ts(result["submitted_at"]) - pushed[sid]["pushed"])
    if intake:
        intake.sort()
        print(f"intake latency (push → enqueued): median {statistics.median(intake):.0f}s, "
              f"p95 {intake[int(len(intake) * 0.95) - 1]:.0f}s, max {intake[-1]:.0f}s")

    # Per-team cap: at no moment more than one stress job between enqueue and end.
    worst = {}
    for sid, result in results.items():
        team = result["team_id"]
        start, end = _ts(result.get("submitted_at")), _ts(result.get("finished_at"))
        if start and end:
            worst.setdefault(team, []).append((start, end))
    for sid, (state, job) in jobs.items():
        if state in ("pending", "claimed"):
            worst.setdefault(job["team_id"], []).append((_ts(job["enqueued_at"]), time.time()))
    for team, spans in sorted(worst.items()):
        events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
        level = peak = 0
        for _, step in events:
            level += step
            peak = max(peak, level)
        print(f"{team}: {len(spans)} jobs, max active at once {peak} (cap 1)")

    board = root / "state" / "leaderboard.json"
    if board.exists():
        data = json.loads(board.read_text())
        rows = sum(len(t.get("activity", [])) for t in data["queue"]["teams"])
        print(f"leaderboard.json: {board.stat().st_size / 1024:.0f} KB, {rows} history rows, "
              f"generated {data.get('generated_at')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["setup", "run", "report"])
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--teams", type=int, default=3)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--tag", default=datetime.now(UTC).strftime("%m%d"))
    parser.add_argument("--config", default="~/capx-queue/smoke/smoke.yaml")
    args = parser.parse_args()
    {"setup": setup, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
