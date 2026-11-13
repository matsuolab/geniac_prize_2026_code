"""How the runner wires a run up — especially what the submission can and cannot reach.

``singularity`` is not available in a GPU-free test environment, so ``_spawn`` is
replaced with a fake that records the command lines and plays the part of each
process (creating the sockets, writing the gateway's results dump). The commands
are the thing under test: they *are* the isolation.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from capx.evalqueue import runner as runner_module
from capx.evalqueue import scratch
from capx.evalqueue.config import (
    QueueConfig,
    RemoteBackendConfig,
    SimulatorConfig,
    SubsetConfig,
)
from capx.evalqueue.leaderboard import score_result
from capx.evalqueue.models import ErrorCode, QueueJob, RunStatus, WeightsMount
from capx.evalqueue.runner import (
    EvalRunner,
    Interrupted,
    LeaseLost,
    SubsetFailed,
    _read_leaderboard,
    _simulators_for,
)
from capx.evalqueue.store import EvalStore, utcnow
from capx.protocol.taskpack import load_pack

PACK_LIMITS = {"trial_time_s": 60.0, "n_trials": 1, "trial_steplimit": 100}
LEADERBOARD = {
    "visibility": "public", "overall_success_rate": 0.5, "successes": 1, "trials": 2, "tasks": {"t": {}}
}


class FakePopen:
    def __init__(self, *, polls_before_exit: int = 0) -> None:
        self.returncode: int | None = None
        self.signals: list[int] = []
        self.hangs = False
        self.code_on_signal = 0
        self._polls_before_exit = polls_before_exit

    def poll(self) -> int | None:
        if self._polls_before_exit > 0:
            self._polls_before_exit -= 1
            return None
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.hangs:
            raise subprocess.TimeoutExpired(cmd="submission", timeout=timeout or 0)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)
        self.hangs = False
        self._polls_before_exit = 0
        self.returncode = self.code_on_signal

    def kill(self) -> None:
        self.signals.append(signal.SIGKILL)
        self.hangs = False
        self._polls_before_exit = 0
        self.returncode = -9


class FakeSpawn:
    """Stands in for the three container launches, and records their command lines."""

    def __init__(
        self,
        *,
        submission_exit: int = 0,
        write_results: bool = True,
        submission_timeout: bool = False,
        connects: bool = True,
        finishes_after_s: float | None = None,
        lanes: dict[str, int] | None = None,
        exits_late: bool = False,
        runs_on: bool = False,
        first_contact_in_s: float = 0.0,
        finish_taken_after_s: float | None = None,
    ) -> None:
        """``exits_late``: exits on its own, after its budget ran out. ``runs_on``: keeps
        running until it is stopped. ``finish_taken_after_s``: the gateway's dump says it
        took a finish() then, from the first contact, though no job-finished file came."""
        self.commands: dict[str, list[str]] = {}
        self.log_caps: dict[str, int | None] = {}
        self.procs: dict[str, runner_module._Proc] = {}
        self._lanes = lanes
        self._submission_exit = submission_exit
        self._write_results = write_results
        self._submission_timeout = submission_timeout
        self._connects = connects
        self._finishes_after_s = finishes_after_s
        self._exits_late = exits_late
        self._runs_on = runs_on
        self._first_contact_in_s = first_contact_in_s
        self._finish_taken_after_s = finish_taken_after_s

    def __call__(
        self, name: str, command: list[str], log: Path, *, log_cap: int | None = None
    ) -> runner_module._Proc:
        self.commands[name] = command
        self.log_caps[name] = log_cap
        # A running submission is polled at least once, which is when the runner
        # starts the backends the gateway asked for.
        popen = FakePopen(polls_before_exit=1 if name == "submission" else 0)
        if name.startswith("env-") or name == "gateway":
            _touch_socket(command)
        if name.startswith("forward-"):
            local = Path(command[command.index("-L") + 1].split(":")[0])
            local.parent.mkdir(parents=True, exist_ok=True)
            local.touch()
        if name == "gateway" and self._write_results:
            results = _env_value(command, "CAPX_RESULTS_PATH")
            Path(results).parent.mkdir(parents=True, exist_ok=True)
            document: dict[str, object] = {"leaderboard": LEADERBOARD, "scores": []}
            if self._finish_taken_after_s is not None:
                document["finished_at"] = time.time() + self._finish_taken_after_s
            Path(results).write_text(json.dumps(document))
        if name == "gateway" and self._connects:
            # A hung submission connected long enough ago that its budget is spent.
            spent = self._submission_timeout or self._exits_late
            ago = 2 * PACK_LIMITS["trial_time_s"] if spent else 0.0
            started = time.time() - ago + self._first_contact_in_s
            Path(_env_value(command, "CAPX_CLOCK_STARTED_PATH")).write_text(f"{started}\n")
            if self._finishes_after_s is not None:
                Path(_env_value(command, "CAPX_JOB_FINISHED_PATH")).write_text(
                    f"{started + self._finishes_after_s}\n"
                )
        if name == "submission" and self._connects:
            self._ask_for_backends()
        if name == "submission":
            if self._submission_timeout:
                popen.hangs = True
            elif self._exits_late:
                popen = FakePopen()
                popen.returncode = 0
            elif self._runs_on:
                popen = FakePopen(polls_before_exit=10**6)
                popen.code_on_signal = -15
            elif self._finishes_after_s is not None:
                # Called finish() and keeps running until it is stopped.
                popen = FakePopen(polls_before_exit=10**6)
                popen.code_on_signal = -15
            else:
                popen.returncode = self._submission_exit
        self.procs[name] = runner_module._Proc(name=name, popen=popen, log=log)
        return self.procs[name]


    def _ask_for_backends(self) -> None:
        """What the gateway does as the submission opens lanes: one per simulator,
        unless the test says how many."""
        gateway = self.commands["gateway"]
        seen: dict[str, int] = {}
        for route in _env_value(gateway, "CAPX_BACKENDS").split(","):
            simulator, _, address = route.partition("=unix:")
            seen[simulator] = seen.get(simulator, 0) + 1
            if seen[simulator] <= (self._lanes or {}).get(simulator, 1):
                want = _host_path(gateway, address)
                want.with_name(want.name + ".want").touch()


def _env(command: list[str]) -> dict[str, str]:
    """Read back the env file the runner wrote for this container, as a shell would."""
    for index, token in enumerate(command):
        if token == "--env-file":
            words = shlex.split(Path(command[index + 1]).read_text())
            return dict(word.split("=", 1) for word in words)
    raise AssertionError(f"no --env-file in {command}")


def _env_value(command: list[str], key: str) -> str:
    env = _env(command)
    if key not in env:
        raise AssertionError(f"{key} not passed in {command}")
    return env[key]


def _binds(command: list[str]) -> list[str]:
    return [command[i + 1] for i, token in enumerate(command) if token == "--bind"]


def _host_path(command: list[str], inside: str) -> Path:
    """Translate a container path back through the command's --bind mappings.

    The sockets are named by their in-container path (organizer containers get a
    private /tmp, so the host path is not visible to them), and a test that wants
    to touch or compare one has to undo that mapping.
    """
    for bind in _binds(command):
        source, _, destination = bind.partition(":")
        destination = destination or source
        if inside == destination or inside.startswith(f"{destination}/"):
            return Path(source) / inside[len(destination) :].lstrip("/")
    raise AssertionError(f"{inside} is not bound in {command}")


def _touch_socket(command: list[str]) -> None:
    for key in ("CAPX_BACKEND_UNIX", "CAPX_UNIX"):
        try:
            inside = _env_value(command, key)
        except AssertionError:
            continue
        path = _host_path(command, inside)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("capx-gateway.sif", "capx-env-robosuite.sif"):
        (images / name).write_bytes(b"sif")

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pack.yaml").write_text(
        yaml.safe_dump(
            {
                "pack_id": "p",
                "version": "1",
                "visibility": "public",
                "limits": PACK_LIMITS,
                "tasks": [
                    {
                        "task_id": "t",
                        "embodiment": "franka",
                        "simulator": "robosuite",
                        "instruction": "go",
                        "camera_names": ["c"],
                        "env_target": "fake",
                        "seed_policy": {"mode": "fixed", "seeds": [1]},
                    }
                ],
            }
        )
    )

    config = QueueConfig(
        root=str(tmp_path / "root"),
        master_seed=1,
        socket_dir=str(tmp_path),
        subsets=[SubsetConfig(name="public", taskpack=str(pack))],
    )
    config.gateway_image = str(images / "capx-gateway.sif")
    config.simulators = {
        "robosuite": SimulatorConfig(image=str(images / "capx-env-robosuite.sif"), max_lanes=1)
    }
    monkeypatch.setattr(runner_module.time, "sleep", lambda _s: None)

    store = EvalStore(config.root)
    store.ensure_layout()

    image = tmp_path / "agent.sif"
    image.write_bytes(b"team image")
    job = QueueJob(
        job_id="alpha__s1",
        team_id="alpha",
        submission_id="s1",
        image_path=str(image),
        enqueued_at=utcnow(),
    )
    return config, store, job, monkeypatch


def test_a_successful_run_returns_the_gateways_leaderboard(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)

    result = EvalRunner(config, store).run(job, "run1")

    assert result.status is RunStatus.COMPLETED
    assert result.subsets["public"].status == "completed"
    assert result.subsets["public"].leaderboard == LEADERBOARD
    # Time on the referee's clock, first contact to submission exit, within the budget.
    used = result.subsets["public"].budget_used_s
    assert used is not None and 0 <= used <= PACK_LIMITS["trial_time_s"]
    assert result.subsets["public"].over_time is False


def test_the_submission_gets_no_network(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    command = spawn.commands["submission"]
    for flag in ("--net", "--network=none", "--containall"):
        assert flag in command, f"{flag} missing: the submission would not be isolated"


def test_the_submission_cannot_see_the_backend_sockets(setup) -> None:
    """The backend link carries raw obs and reward — compose keeps it on a separate
    network, and here it must stay outside the submission's mounts."""
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    backend = spawn.commands["env-robosuite-0"]
    backend_socket = _host_path(backend, _env_value(backend, "CAPX_BACKEND_UNIX"))
    mounted = [Path(bind.split(":")[0]) for bind in _binds(spawn.commands["submission"])]
    assert not any(
        source == backend_socket.parent or source in backend_socket.parents for source in mounted
    ), f"{backend_socket} is reachable through {mounted}"

    gateway = spawn.commands["gateway"]
    gateway_socket = _host_path(gateway, _env_value(gateway, "CAPX_UNIX"))
    assert gateway_socket.parent in mounted  # ...while the gateway still is
    # Read-only: the directory is the node's /tmp, which the other worker's sockets share.
    assert f"{gateway_socket.parent}:/capx/run:ro" in _binds(spawn.commands["submission"])


def test_the_submission_gets_a_writable_home(setup) -> None:
    """--containall's session HOME is tiny, and libraries that cache without asking
    (Triton compiles kernels into $HOME/.triton) fail with ENOSPC on it."""
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    command = spawn.commands["submission"]
    # Must be --home: Singularity refuses to let --env set HOME.
    assert "--home" in command
    source, _, destination = command[command.index("--home") + 1].partition(":")
    assert Path(source).is_dir()
    assert destination == runner_module.CONTAINER_HOME
    assert _env_value(command, "XDG_CACHE_HOME").startswith(destination)


def test_a_submission_can_set_its_own_environment_but_not_the_organizers(setup) -> None:
    config, store, job, monkeypatch = setup
    job = job.model_copy(
        update={
            "env": {
                "MY_MODEL": "some/model",
                "CAPX_ENV_SOCKET": "/tmp/not-the-gateway.sock",
                "HF_HUB_OFFLINE": "0",
            }
        }
    )
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    command = spawn.commands["submission"]
    assert _env_value(command, "MY_MODEL") == "some/model"
    assert _env_value(command, "CAPX_ENV_SOCKET") == "/capx/run/gateway.sock"
    assert _env_value(command, "HF_HUB_OFFLINE") == "1"


def test_team_env_values_arrive_verbatim_and_cannot_add_variables(setup) -> None:
    """Singularity reads the env file as shell: an unquoted space kept the container
    from starting, and a newline in a value added a line of its own."""
    config, store, job, monkeypatch = setup
    args = "--attention-backend TRITON_ATTN --max-num-seqs 64 'x' \"y\" #z"
    job = job.model_copy(
        update={"env": {"ARGS": args, "ZZZ": "x\nCAPX_ENV_SOCKET=/tmp/evil.sock\nHF_HUB_OFFLINE=0"}}
    )
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    env = _env(spawn.commands["submission"])
    assert env["ARGS"] == args
    assert env["ZZZ"].startswith("x\n")
    assert env["CAPX_ENV_SOCKET"] == "/capx/run/gateway.sock"
    assert env["HF_HUB_OFFLINE"] == "1"


@pytest.mark.parametrize("name", ["ZZZ=x\nCAPX_ENV_SOCKET", "A=B", "A B", "1A", "A$(id)", ""])
def test_an_env_name_that_is_not_a_plain_variable_name_is_never_written(tmp_path, name) -> None:
    path = tmp_path / "submission.env"
    with pytest.raises(SubsetFailed, match="environment variable names") as failure:
        runner_module._write_env_file(path, {"OK_NAME": "v", name: "v"})
    assert failure.value.code is ErrorCode.INTERNAL_ERROR
    assert not path.exists()


def test_a_run_with_an_unsafe_env_name_fails_on_our_side_before_starting(setup) -> None:
    config, store, job, monkeypatch = setup
    job = job.model_copy(update={"env": {"ZZZ=x\nCAPX_ENV_SOCKET": "/tmp/evil.sock"}})
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)

    result = EvalRunner(config, store).run(job, "run1")

    assert result.error_code is ErrorCode.INTERNAL_ERROR
    assert "submission" not in spawn.commands
    assert score_result(result, ["public"]) == (None, 0)


def test_started_commands_get_none_of_the_workers_secrets(tmp_path, monkeypatch) -> None:
    """Singularity expands an env file's variables with its own environment, so the
    worker's tokens must not be in it."""
    monkeypatch.setenv("CAPX_HF_TOKEN", "hf_organizer")
    monkeypatch.setenv("SINGULARITY_BIND", "/etc")
    log = tmp_path / "probe.log"
    proc = runner_module._spawn("probe", ["env"], log)
    proc.popen.wait(timeout=30)

    names = {line.split("=", 1)[0] for line in log.read_text().splitlines()[2:] if "=" in line}
    assert "PATH" in names
    assert not names - set(runner_module.HOST_ENV)


def test_the_submission_log_keeps_its_head_and_tail_only(tmp_path) -> None:
    log = tmp_path / "submission.log"
    script = "import sys; sys.stdout.write('HEAD' + 'x' * 100_000 + 'TAIL')"
    proc = runner_module._spawn("probe", [sys.executable, "-c", script], log, log_cap=1000)
    proc.popen.wait(timeout=30)
    runner_module._stop(proc)

    text = log.read_text()
    output = text.split("\n\n", 1)[1]
    assert output.startswith("HEAD") and output.endswith("TAIL")
    assert "bytes of output left out here" in output
    assert len(output) < 1000 + (1 << 16) + 100


def test_the_submission_log_is_kept_whole_under_the_cap(tmp_path) -> None:
    log = tmp_path / "submission.log"
    script = "print('line one'); print('line two')"
    proc = runner_module._spawn("probe", [sys.executable, "-c", script], log, log_cap=1000)
    proc.popen.wait(timeout=30)
    runner_module._stop(proc)
    assert log.read_text().split("\n\n", 1)[1] == "line one\nline two\n"


def test_the_submission_log_is_capped(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")
    assert spawn.log_caps["submission"] == runner_module.SUBMISSION_LOG_BYTES


def test_declared_weights_are_mounted_read_only_and_nothing_else_is(setup) -> None:
    config, store, job, monkeypatch = setup
    mount = WeightsMount(repo="org/model", revision="a" * 40, path="/models/m")
    store.weights_dir(mount).mkdir(parents=True)
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job.model_copy(update={"weights": [mount]}), "run1")

    command = spawn.commands["submission"]
    assert f"{store.weights_dir(mount)}:/models/m:ro" in _binds(command)
    assert not any(str(store.weights) == bind.split(":")[0] for bind in _binds(command))
    assert "HF_HOME" not in _env(command)


def test_declared_weights_missing_from_the_cache_fail_the_run_before_anything_starts(
    setup,
) -> None:
    config, store, job, monkeypatch = setup
    mount = WeightsMount(repo="org/model", revision="a" * 40, path="/models/m")
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    result = EvalRunner(config, store).run(job.model_copy(update={"weights": [mount]}), "run1")

    assert result.status is RunStatus.FAILED
    assert "org/model" in (result.error or "")
    assert spawn.commands == {}


def test_every_container_is_pinned_to_a_working_directory(setup) -> None:
    """Without --pwd a container inherits the host's cwd, and a relative script path
    then runs the host checkout instead of the code baked into the image."""
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    for name, command in spawn.commands.items():
        assert "--pwd" in command, f"{name} runs with whatever cwd it inherited"
    for name in ("gateway", "env-robosuite-0"):
        command = spawn.commands[name]
        assert command[command.index("--pwd") + 1] == runner_module.IMAGE_WORKDIR
        assert command[-1].startswith(runner_module.IMAGE_WORKDIR)


def test_the_gateway_runs_the_configured_pack_in_official_mode(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    command = spawn.commands["gateway"]
    assert _env_value(command, "CAPX_MODE") == "official"
    assert _env_value(command, "CAPX_SIMULATORS") == "robosuite"
    assert _env_value(command, "CAPX_BACKENDS").startswith("robosuite=unix:")


def test_the_env_backend_reads_the_pack_it_builds_envs_for(setup) -> None:
    """A pack may ship its own env code, which only the backend imports."""
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    taskpack = Path(config.subsets[0].taskpack)
    inside = f"{runner_module.IMAGE_WORKDIR}/taskpacks/{taskpack.name}"
    command = spawn.commands["env-robosuite-0"]
    assert f"{taskpack.resolve()}:{inside}:ro" in _binds(command)
    assert _env_value(command, "CAPX_TASKPACK") == f"taskpacks/{taskpack.name}"


def test_a_pack_gets_every_directory_it_needs_mounted(setup) -> None:
    """The extended pack holds the tasks and the shared dir the env code, so both
    containers need them."""
    config, store, job, monkeypatch = setup
    base = Path(config.subsets[0].taskpack)
    middle = base.parent / "middle"
    middle.mkdir()
    (middle / "pack.yaml").write_text(
        yaml.safe_dump(
            {"pack_id": "middle", "limits": PACK_LIMITS, "extends": base.name, "task_ids": ["t"]}
        )
    )
    shared = base.parent / "shared"
    shared.mkdir()
    document = yaml.safe_load((base / "pack.yaml").read_text())
    (base / "pack.yaml").write_text(yaml.safe_dump({**document, "code_dirs": ["shared"]}))
    config.subsets[0].taskpack = str(middle)
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    for name in ("gateway", "env-robosuite-0"):
        binds = _binds(spawn.commands[name])
        for pack in (middle, base, shared):
            inside = f"{runner_module.IMAGE_WORKDIR}/taskpacks/{pack.name}"
            assert f"{pack.resolve()}:{inside}:ro" in binds, name


def test_backends_start_only_for_the_lanes_the_submission_uses(setup) -> None:
    """Two lanes cannot share a backend process: the factory inside it caches one
    simulator, and they would race on its physics and render state. So each lane
    gets one — but only once the gateway asks for it, so a cap of 3 with 2 lanes in
    use holds 2 processes, not 3."""
    config, store, job, monkeypatch = setup
    config.simulators["robosuite"].max_lanes = 3
    spawn = FakeSpawn(lanes={"robosuite": 2})
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    backends = [name for name in spawn.commands if name.startswith("env-robosuite")]
    assert sorted(backends) == ["env-robosuite-0", "env-robosuite-1"]
    sockets = {_env_value(spawn.commands[name], "CAPX_BACKEND_UNIX") for name in backends}
    assert len(sockets) == 2

    gateway = spawn.commands["gateway"]
    assert len(_env_value(gateway, "CAPX_BACKENDS").split(",")) == 3  # the cap
    assert _env_value(gateway, "CAPX_BACKENDS_ON_DEMAND") == "1"
    assert _env_value(spawn.commands["submission"], "CAPX_MAX_LANES") == "3"


def test_a_simulator_needs_a_backend_configured(setup) -> None:
    config, store, job, monkeypatch = setup
    config.simulators = {}
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn())
    result = EvalRunner(config, store).run(job, "run1")
    assert "no backend configured" in (result.subsets["public"].error or "")


def test_a_multi_simulator_route_list_survives_intact(setup) -> None:
    """Singularity's --env splits values on commas, which silently dropped every
    backend route but the first; the env file must carry all of them."""
    config, store, job, monkeypatch = setup
    config.subsets[0].simulators = []  # take every simulator in the pack
    pack = Path(config.subsets[0].taskpack) / "pack.yaml"
    document = yaml.safe_load(pack.read_text())
    second = dict(document["tasks"][0], task_id="t2", simulator="libero")
    document["tasks"].append(second)
    pack.write_text(yaml.safe_dump(document))
    config.simulators["libero"] = config.simulators["robosuite"]

    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    backends = _env_value(spawn.commands["gateway"], "CAPX_BACKENDS")
    assert backends.count("unix:") == 2, backends
    assert _env_value(spawn.commands["gateway"], "CAPX_SIMULATORS") == "libero,robosuite"


def test_a_nonzero_submission_exit_fails_the_subset_but_keeps_its_scores(setup) -> None:
    """A submission that scored two trials and then crashed should keep those two;
    discarding them would punish the team for the tail rather than the part that
    worked, and would waste the GPU time that produced them."""
    config, store, job, monkeypatch = setup
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn(submission_exit=3))

    result = EvalRunner(config, store).run(job, "run1")
    assert result.status is RunStatus.FAILED
    assert "exited with code 3" in result.subsets["public"].error
    assert result.subsets["public"].error_code is ErrorCode.SUBMISSION_FAILED
    assert result.error_code is ErrorCode.SUBMISSION_FAILED
    assert result.subsets["public"].leaderboard == LEADERBOARD


def test_a_missing_results_dump_fails_the_subset_rather_than_the_worker(setup) -> None:
    config, store, job, monkeypatch = setup
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn(write_results=False))

    result = EvalRunner(config, store).run(job, "run1")
    assert result.status is RunStatus.FAILED
    assert "no results" in result.subsets["public"].error
    assert result.subsets["public"].error_code is ErrorCode.NO_SCORE


def test_a_submission_that_runs_out_of_time_keeps_what_it_scored(setup) -> None:
    """The runner stops a submission once the pack's job budget, counted from its
    first contact with the gateway, is spent — even one that wedged without calling
    the gateway, as a real agent stuck in its own retry backoff did. Whatever the
    gateway scored before that still counts."""
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn(submission_timeout=True)
    monkeypatch.setattr(runner_module, "_spawn", spawn)

    result = EvalRunner(config, store).run(job, "run1")
    assert result.status is RunStatus.TIMEOUT
    assert result.subsets["public"].status == "timeout"
    assert "job budget" in (result.subsets["public"].error or "")
    assert result.subsets["public"].leaderboard == LEADERBOARD
    assert result.subsets["public"].over_time is True
    assert result.subsets["public"].budget_used_s == PACK_LIMITS["trial_time_s"]


def test_a_submission_that_writes_too_much_is_stopped_and_scored_as_if_it_exited(setup) -> None:
    """Rulebook §5: /capx/out and HOME are on the shared root, so a submission over the
    limit is stopped. That ends the job as an exit would (A.1), and what the gateway
    recorded until then is scored. Nothing after that counts: the gateway is told to
    end the job (it does on SIGTERM, before its final dump), and the submission is
    killed with no grace period to go on in."""
    config, store, job, monkeypatch = setup
    config.limits.submission_storage_files = 5
    monkeypatch.setattr(scratch, "CHECK_INTERVAL_S", 0.0)
    spawn = FakeSpawn(runs_on=True)

    def spawn_and_write(name, command, log, **kwargs):
        proc = spawn(name, command, log, **kwargs)
        if name == "submission":
            for index in range(10):
                (_host_path(command, "/capx/out") / f"frame-{index}.png").write_bytes(b"x")
        return proc

    monkeypatch.setattr(runner_module, "_spawn", spawn_and_write)
    result = EvalRunner(config, store).run(job, "run1")

    subset = result.subsets["public"]
    assert subset.status is RunStatus.FAILED and result.status is RunStatus.FAILED
    assert subset.error_code is result.error_code is ErrorCode.SUBMISSION_STORAGE_EXCEEDED
    assert "/capx/out and HOME" in (subset.error or "")
    assert subset.leaderboard == LEADERBOARD
    assert subset.budget_used_s is not None and subset.over_time is False
    assert score_result(result, ["public"]) == (0.5, 1)
    assert spawn.procs["submission"].popen.signals == [signal.SIGKILL]
    assert signal.SIGTERM in spawn.procs["gateway"].popen.signals


class _OverTheLimit:
    """Stands in for StorageLimit: the trees were found over the limit at ``at``."""

    def __init__(self, at: datetime) -> None:
        self.exceeded = ("submission wrote more than it may", at)

    def __call__(self, *_args, **_kwargs) -> _OverTheLimit:
        return self

    def poll(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("finish_after_s", "status"),
    [(-60.0, RunStatus.COMPLETED), (30.0, RunStatus.FAILED)],
    ids=["finish-first", "finish-after"],
)
def test_only_a_finish_before_the_limit_was_found_exceeded_counts(
    setup, finish_after_s, status
) -> None:
    """A finish() at or before the moment the limit was found exceeded ended the job
    first. One that lands later, as it can from a gateway that has not ended the job yet,
    does not undo the stop."""
    config, store, job, monkeypatch = setup
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn(finishes_after_s=finish_after_s))
    # Five seconds into the job: the fake gateway's clock starts as the run does.
    monkeypatch.setattr(runner_module, "StorageLimit", _OverTheLimit(utcnow() + timedelta(seconds=5)))

    result = EvalRunner(config, store).run(job, "run1")

    subset = result.subsets["public"]
    assert subset.status is status
    if status is RunStatus.FAILED:
        assert subset.error_code is ErrorCode.SUBMISSION_STORAGE_EXCEEDED
        assert subset.budget_used_s == pytest.approx(5.0, abs=1.0)
    else:
        assert subset.error_code is None


@pytest.mark.parametrize(
    ("taken_after_s", "status"),
    [(2.0, RunStatus.COMPLETED), (8.0, RunStatus.FAILED)],
    ids=["taken-first", "taken-after"],
)
def test_a_finish_the_gateway_took_first_counts_however_late_its_file_comes(
    setup, taken_after_s, status
) -> None:
    """The referee took the finish() 2 s into the job and the limit was found exceeded
    at 5 s, but the job-finished file was still being written then (here: never came).
    The gateway's final dump says when it took it."""
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn(runs_on=True, finish_taken_after_s=taken_after_s)
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    monkeypatch.setattr(runner_module, "StorageLimit", _OverTheLimit(utcnow() + timedelta(seconds=5)))

    result = EvalRunner(config, store).run(job, "run1")

    subset = result.subsets["public"]
    assert subset.status is status
    if status is RunStatus.COMPLETED:
        assert subset.error_code is None
        assert subset.budget_used_s == pytest.approx(2.0, abs=1.0)
    else:
        assert subset.error_code is ErrorCode.SUBMISSION_STORAGE_EXCEEDED
        assert subset.budget_used_s == pytest.approx(5.0, abs=1.0)


def test_a_submission_that_calls_finish_is_stopped_and_timed_to_the_finish(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn(finishes_after_s=42.0)
    monkeypatch.setattr(runner_module, "_spawn", spawn)

    result = EvalRunner(config, store).run(job, "run1")

    # Stopped by us, so the signal's exit code is not a failure.
    assert result.status is RunStatus.COMPLETED
    assert result.subsets["public"].leaderboard == LEADERBOARD
    assert result.subsets["public"].budget_used_s == 42.0


def test_a_submission_that_exits_after_its_budget_ran_out_is_over_time(setup) -> None:
    """The referee refuses every call once the budget is gone, and a submission that
    handles that and exits 0 still completes. Whether it ran out of time comes from
    the gateway's clock, not from what it printed."""
    config, store, job, monkeypatch = setup
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn(exits_late=True))

    result = EvalRunner(config, store).run(job, "run1")

    subset = result.subsets["public"]
    assert result.status is RunStatus.COMPLETED
    assert subset.over_time is True
    assert subset.budget_used_s == PACK_LIMITS["trial_time_s"]


def test_what_the_submission_prints_does_not_make_it_over_time(setup) -> None:
    config, store, job, monkeypatch = setup
    log = store.run_dir("run1") / "public" / "submission.log"
    log.parent.mkdir(parents=True)
    log.write_text("GatewayError: job_timeout: job budget exhausted\n")
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn())

    result = EvalRunner(config, store).run(job, "run1")
    assert result.subsets["public"].over_time is False


def test_the_submission_is_stopped_whatever_ends_the_wait(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn(runs_on=True)
    monkeypatch.setattr(runner_module, "_spawn", spawn)

    def broken_poll(_backends) -> None:
        raise OSError("the shared filesystem went away")

    monkeypatch.setattr(runner_module._Backends, "poll", broken_poll)

    result = EvalRunner(config, store).run(job, "run1")

    assert result.error_code is ErrorCode.INTERNAL_ERROR
    assert signal.SIGTERM in spawn.procs["submission"].popen.signals


def test_an_interrupted_run_stops_everything_and_raises(setup) -> None:
    """A worker that Slurm is stopping does not publish the run (worker.py)."""
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn(runs_on=True)
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    runner = EvalRunner(config, store)
    monkeypatch.setattr(runner_module._Backends, "poll", lambda _backends: runner.interrupt())

    with pytest.raises(Interrupted):
        runner.run(job, "run1")
    assert signal.SIGTERM in spawn.procs["submission"].popen.signals
    assert signal.SIGTERM in spawn.procs["gateway"].popen.signals

    spawn.commands.clear()
    with pytest.raises(Interrupted):
        runner.run(job, "run2")
    assert spawn.commands == {}


def test_a_run_past_the_deadline_is_cancelled_without_starting(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    config.deadline = utcnow() - timedelta(seconds=1)

    result = EvalRunner(config, store).run(job, "run1")

    assert result.status is RunStatus.CANCELLED and result.subsets == {}
    assert spawn.commands == {}


def test_a_run_going_at_the_deadline_is_stopped_and_cancelled(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    looks = iter([False, True])  # open when the subset starts, closed at the first tick
    monkeypatch.setattr(runner_module, "past_deadline", lambda _config, now=None: next(looks, True))

    result = EvalRunner(config, store).run(job, "run1")

    # Not scored even though the gateway wrote results: the round is over.
    assert result.status is RunStatus.CANCELLED and result.subsets == {}
    assert "deadline" in (result.error or "")


def _closed_after(deadline):
    """``past_deadline`` for a round that is open whenever the runner looks on its own,
    so only what the submission's end is compared with is the real deadline."""
    return lambda _config, now=None: now is not None and now >= deadline


def test_a_submission_that_exits_after_the_deadline_between_ticks_is_cancelled(setup) -> None:
    config, store, job, monkeypatch = setup
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn())
    monkeypatch.setattr(runner_module, "past_deadline", _closed_after(utcnow() - timedelta(seconds=1)))
    result = EvalRunner(config, store).run(job, "run1")
    assert result.status is RunStatus.CANCELLED and result.subsets == {}


def test_a_finish_before_the_deadline_counts_even_if_seen_after_it(setup) -> None:
    config, store, job, monkeypatch = setup
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn(finishes_after_s=-60.0))
    monkeypatch.setattr(runner_module, "past_deadline", _closed_after(utcnow() - timedelta(seconds=30)))
    result = EvalRunner(config, store).run(job, "run1")
    assert result.status is RunStatus.COMPLETED


class _ClosesAt:
    """``past_deadline`` for a round that closes at the runner's ``n``-th own look. The
    look at when the submission ended always finds it open."""

    def __init__(self, n: int) -> None:
        self.looks, self._n = 0, n

    def __call__(self, _config, now=None) -> bool:
        if now is not None:
            return False
        self.looks += 1
        return self.looks >= self._n


def test_a_deadline_during_gateway_startup_cancels_before_the_submission_starts(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    # Open before the subset and after the packs are copied; closed once the gateway is up.
    monkeypatch.setattr(runner_module, "past_deadline", _ClosesAt(3))
    result = EvalRunner(config, store).run(job, "run1")
    assert result.status is RunStatus.CANCELLED
    assert "submission" not in spawn.commands


CLOSE = datetime(2030, 1, 1, tzinfo=UTC)
"""A deadline far enough ahead that the real clock never reaches it in a test."""


@pytest.mark.parametrize(
    ("failed_at", "status"),
    [(CLOSE + timedelta(seconds=1), RunStatus.CANCELLED), (CLOSE - timedelta(minutes=1), RunStatus.FAILED)],
    ids=["after", "before"],
)
def test_a_failure_is_cancelled_only_if_it_came_after_the_deadline(
    setup, failed_at, status
) -> None:
    """Teardown can run past the deadline; what counts is when the run went wrong."""
    config, store, job, monkeypatch = setup
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn(submission_exit=3))
    monkeypatch.setattr(runner_module, "utcnow", lambda: failed_at)
    config.deadline = CLOSE
    result = EvalRunner(config, store).run(job, "run1")
    assert result.status is status
    if status is RunStatus.FAILED:
        assert result.subsets["public"].leaderboard == LEADERBOARD  # scored, not discarded


def test_a_timeout_before_the_deadline_keeps_its_scores_however_long_the_stop_takes(setup) -> None:
    config, store, job, monkeypatch = setup
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn(submission_timeout=True))
    clock = [CLOSE - timedelta(seconds=1)]
    monkeypatch.setattr(runner_module, "utcnow", lambda: clock[0])
    stop = runner_module._stop

    def slow_stop(proc, grace_s=30.0):
        clock[0] = CLOSE + timedelta(seconds=30)  # SIGTERM, then the grace period
        stop(proc, grace_s)

    monkeypatch.setattr(runner_module, "_stop", slow_stop)
    config.deadline = CLOSE
    result = EvalRunner(config, store).run(job, "run1")
    assert result.status is RunStatus.TIMEOUT
    assert result.subsets["public"].leaderboard == LEADERBOARD


def test_a_submission_that_never_connects_is_stopped_and_not_scored(setup) -> None:
    """Rulebook §4.3: a submission that does not reach the gateway within the startup
    limit is not evaluated. The gateway's first dump (0 of N) is no score, and the job
    clock never started, so it did not run out of time either."""
    config, store, job, monkeypatch = setup
    config.limits.submission_startup_s = 0.01
    spawn = FakeSpawn(submission_timeout=True, connects=False)
    monkeypatch.setattr(runner_module, "_spawn", spawn)

    result = EvalRunner(config, store).run(job, "run1")

    subset = result.subsets["public"]
    assert "did not reach the gateway" in (subset.error or "")
    assert subset.status is RunStatus.FAILED and result.status is RunStatus.FAILED
    assert subset.error_code is result.error_code is ErrorCode.SUBMISSION_STARTUP_TIMEOUT
    assert subset.leaderboard is None and subset.budget_used_s is None
    assert subset.over_time is False
    assert score_result(result, ["public"]) == (None, 0)
    assert signal.SIGTERM in spawn.procs["submission"].popen.signals


def test_a_submission_that_exits_after_the_startup_limit_without_contact_is_not_scored(
    setup,
) -> None:
    """It exits between two looks, so the wait ends on the exit, not on the limit."""
    config, store, job, monkeypatch = setup
    config.limits.submission_startup_s = 0.2
    spawn = FakeSpawn(connects=False)

    class ExitsAfterTheLimit(FakePopen):
        def poll(self) -> int | None:
            if self.returncode is None:
                self.returncode = 0
                return None  # still running at the first look
            threading.Event().wait(0.3)  # gone by the next, which is past the limit
            return self.returncode

    def spawn_late_exit(name, command, log, **kwargs):
        proc = spawn(name, command, log, **kwargs)
        if name == "submission":
            proc.popen = ExitsAfterTheLimit()
        return proc

    monkeypatch.setattr(runner_module, "_spawn", spawn_late_exit)
    result = EvalRunner(config, store).run(job, "run1")

    subset = result.subsets["public"]
    assert subset.error_code is ErrorCode.SUBMISSION_STARTUP_TIMEOUT
    assert subset.leaderboard is None and subset.over_time is False
    assert score_result(result, ["public"]) == (None, 0)


def test_a_late_first_contact_is_not_scored_even_with_a_finish(setup) -> None:
    config, store, job, monkeypatch = setup
    config.limits.submission_startup_s = 30.0
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn(first_contact_in_s=60.0, finishes_after_s=1.0))

    result = EvalRunner(config, store).run(job, "run1")

    assert result.subsets["public"].error_code is ErrorCode.SUBMISSION_STARTUP_TIMEOUT
    assert score_result(result, ["public"]) == (None, 0)


@pytest.mark.parametrize("exits", [False, True], ids=["still-running", "exited-meanwhile"])
def test_a_first_contact_after_the_startup_limit_is_not_scored(setup, exits) -> None:
    """The contact came between two looks, after the limit: the clock file is there,
    but its time is past the cutoff."""
    config, store, job, monkeypatch = setup
    config.limits.submission_startup_s = 30.0
    spawn = FakeSpawn(first_contact_in_s=60.0, runs_on=not exits)

    def spawn_and_maybe_exit(name, command, log, **kwargs):
        proc = spawn(name, command, log, **kwargs)
        if name == "submission" and exits:
            proc.popen = FakePopen()
            proc.popen.returncode = 0
        return proc

    monkeypatch.setattr(runner_module, "_spawn", spawn_and_maybe_exit)
    result = EvalRunner(config, store).run(job, "run1")

    subset = result.subsets["public"]
    assert subset.error_code is ErrorCode.SUBMISSION_STARTUP_TIMEOUT
    assert subset.leaderboard is None
    assert score_result(result, ["public"]) == (None, 0)
    if not exits:
        assert signal.SIGTERM in spawn.procs["submission"].popen.signals


def test_the_gateway_is_told_where_to_record_the_clock_start(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")
    env = _env(spawn.commands["gateway"])
    assert env["CAPX_CLOCK_STARTED_PATH"].endswith("public/clock-started")
    assert env["CAPX_JOB_FINISHED_PATH"].endswith("public/job-finished")
    assert "CAPX_JOB_TIMEOUT_S" not in env
    assert "CAPX_MAX_TRIALS_PER_TASK" not in _env(spawn.commands["submission"])


def test_a_missing_submission_image_fails_the_subset(setup) -> None:
    config, store, job, monkeypatch = setup
    monkeypatch.setattr(runner_module, "_spawn", FakeSpawn())
    Path(job.image_path).unlink()

    result = EvalRunner(config, store).run(job, "run1")
    assert "submission image missing" in result.subsets["public"].error
    assert result.subsets["public"].error_code is ErrorCode.INTERNAL_ERROR


def test_simulators_are_read_from_the_pack_and_filtered(tmp_path) -> None:
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        yaml.safe_dump(
            {
                "pack_id": "p",
                "version": "1",
                "visibility": "public",
                "limits": PACK_LIMITS,
                "tasks": [
                    {
                        "task_id": f"t{i}",
                        "embodiment": embodiment,
                        "simulator": simulator,
                        "instruction": "go",
                        "camera_names": ["c"],
                        "env_target": "fake",
                        "seed_policy": {"mode": "fixed", "seeds": [1]},
                    }
                    for i, (embodiment, simulator) in enumerate(
                        [("franka", "robosuite"), ("franka", "libero"), ("r1pro", "behavior")]
                    )
                ],
            }
        )
    )
    manifest, _ = load_pack(tmp_path)
    everything = SubsetConfig(name="all", taskpack=str(tmp_path))
    assert _simulators_for(manifest, everything) == ["behavior", "libero", "robosuite"]

    subset = SubsetConfig(name="sub", taskpack=str(tmp_path), simulators=["robosuite", "libero"])
    assert _simulators_for(manifest, subset) == ["libero", "robosuite"]


def test_unreadable_results_are_reported_not_swallowed(tmp_path) -> None:
    path = tmp_path / "results.json"
    with pytest.raises(SubsetFailed, match="no results"):
        _read_leaderboard(path)
    path.write_text("{oops")
    with pytest.raises(SubsetFailed, match="not valid JSON"):
        _read_leaderboard(path)
    path.write_text(json.dumps({"scores": []}))
    with pytest.raises(SubsetFailed, match="no leaderboard"):
        _read_leaderboard(path)


class FakeSsh:
    """Stands in for one-shot ssh commands: the remote socket check and the cleanups."""

    def __init__(self, *, not_ready_for: int = 0) -> None:
        self.calls: list[tuple[str, str]] = []
        self._not_ready_for = not_ready_for

    def __call__(self, host: str, remote_command: str, timeout_s: float = 60.0) -> int:
        self.calls.append((host, remote_command))
        if remote_command.startswith("test -S") and self._not_ready_for > 0:
            self._not_ready_for -= 1
            return 1
        return 0


class FakeLease:
    """A remote GPU's lease that is always free, until the test loses it."""

    def __init__(self, key: tuple[str, str, str], on_lost) -> None:
        self.host, self.socket_dir, self.gpu = key
        self.generation = 1
        self._on_lost = on_lost
        self.released = self.lost = False
        self.confirmation: str | None = None
        """What :meth:`confirm` says: the host's answer, None while it agrees."""

    def guard(self, command: str) -> str:
        return f"fence; {command}"

    def confirm(self) -> str | None:
        return self.confirmation

    def fenced(self) -> LeaseLost:
        self.lose()
        return LeaseLost("simbox GPU 0 has been taken over since lease 1")

    def take(self, timeout_s: float = 30.0) -> str | None:
        return None

    def held(self) -> bool:
        return not (self.released or self.lost)

    def release(self) -> None:
        self.released = True

    def lose(self) -> None:
        self.lost = True
        self._on_lost("lost the lease on simbox GPU 0 (ssh exit 255)")


def leased(config: QueueConfig, store: EvalStore) -> EvalRunner:
    """A runner holding its remote GPUs' leases, as the worker hands it a job."""
    runner = EvalRunner(config, store)
    assert runner.take_remote_gpus() is None
    return runner


@pytest.fixture
def remote_setup(setup):
    config, store, job, monkeypatch = setup
    config.simulators = {
        "robosuite": SimulatorConfig(
            remote=RemoteBackendConfig(
                host="simbox", volumes=["assets:/data"], socket_dir="/tmp/capx-remote"
            ),
            max_lanes=1,
        )
    }
    copies: list[tuple[str, list[Path], str]] = []

    def copy_packs(host: str, directories: list[Path], remote_dir: str) -> None:
        copies.append((host, directories, remote_dir))

    monkeypatch.setattr(runner_module, "_copy_packs_to_remote", copy_packs)
    spawn = FakeSpawn()
    spawn.copies = copies
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    monkeypatch.setattr(runner_module, "RemoteGpuLease", FakeLease)
    return config, store, job, monkeypatch, spawn


def test_a_remote_backend_runs_in_docker_without_network(remote_setup) -> None:
    config, store, job, monkeypatch, spawn = remote_setup
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())

    result = leased(config, store).run(job, "run1")

    assert result.status is RunStatus.COMPLETED
    command = spawn.commands["env-robosuite-0"]
    assert command[0] == "ssh" and "simbox" in command
    remote = command[-1]
    assert "docker create" in remote and "--network none" in remote
    assert "-v assets:/data" in remote
    assert 'CAPX_BACKEND_UNIX_OWNER=$(id -u):$(id -g)' in remote


def test_a_remote_backend_gets_the_pack_it_builds_envs_for(remote_setup) -> None:
    """The remote image carries no hidden pack, so the pack goes over with the run."""
    config, store, job, monkeypatch, spawn = remote_setup
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())

    leased(config, store).run(job, "run1")

    taskpack = Path(config.subsets[0].taskpack)
    ((host, copied, remote_dir),) = spawn.copies
    assert host == "simbox" and copied == [taskpack.resolve()]
    remote = spawn.commands["env-robosuite-0"][-1]
    inside = f"{runner_module.IMAGE_WORKDIR}/taskpacks/{taskpack.name}"
    assert f"-v {remote_dir}/taskpacks/{taskpack.resolve().name}:{inside}:ro" in remote
    assert f"-e CAPX_TASKPACK=taskpacks/{taskpack.name}" in remote


def test_a_remote_backend_is_forwarded_to_the_same_route_as_a_local_one(remote_setup) -> None:
    config, store, job, monkeypatch, spawn = remote_setup
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())

    leased(config, store).run(job, "run1")

    gateway = spawn.commands["gateway"]
    assert _env_value(gateway, "CAPX_BACKENDS") == "robosuite=unix:/capx/sock/be/robosuite-0.sock"
    forward = spawn.commands["forward-robosuite-0"]
    local, remote = forward[forward.index("-L") + 1].split(":")
    assert local == str(_host_path(gateway, "/capx/sock/be/robosuite-0.sock"))
    assert remote.startswith("/tmp/capx-remote/") and remote.endswith("/robosuite-0.sock")
    mounted = [Path(bind.split(":")[0]) for bind in _binds(spawn.commands["submission"])]
    assert not any(source in Path(local).parents for source in mounted)


def test_the_forward_waits_until_the_remote_socket_is_ours(remote_setup) -> None:
    config, store, job, monkeypatch, spawn = remote_setup
    ssh = FakeSsh(not_ready_for=2)
    monkeypatch.setattr(runner_module, "_ssh", ssh)

    leased(config, store).run(job, "run1")

    checks = [cmd for _, cmd in ssh.calls if cmd.startswith("test -S")]
    assert len(checks) == 3
    assert " -O " in checks[0]
    assert "forward-robosuite-0" in spawn.commands


def test_teardown_removes_the_remote_container(remote_setup) -> None:
    config, store, job, monkeypatch, spawn = remote_setup
    ssh = FakeSsh()
    monkeypatch.setattr(runner_module, "_ssh", ssh)

    leased(config, store).run(job, "run1")

    container = spawn.commands["env-robosuite-0"][-1].split("--name ")[1].split()[0]
    assert any(cmd.startswith(f"docker rm -f {container}") for _, cmd in ssh.calls)


def test_removing_one_remote_lane_leaves_the_shared_directory(remote_setup) -> None:
    """The remote directory holds every lane's socket and the pack. Retiring one
    lane mid-run must not take the others' sockets with it; the directory goes
    once, when the subset ends."""
    config, store, job, monkeypatch, spawn = remote_setup
    ssh = FakeSsh()
    monkeypatch.setattr(runner_module, "_ssh", ssh)

    leased(config, store).run(job, "run1")

    lane_cleanup = [cmd for _, cmd in ssh.calls if cmd.startswith("docker rm -f")]
    assert lane_cleanup and all("rm -rf" not in cmd for cmd in lane_cleanup)
    assert all(cmd.rstrip().endswith("robosuite-0.sock") for cmd in lane_cleanup)
    dir_cleanup = [cmd for _, cmd in ssh.calls if cmd.startswith("rm -rf /tmp/capx-remote/")]
    assert len(dir_cleanup) == 1


def test_a_remote_backend_that_dies_early_is_removed_without_ending_the_run(remote_setup) -> None:
    """The gateway's wait for its socket times out and voids that episode; the
    rest of the run goes on."""
    config, store, job, monkeypatch, spawn = remote_setup
    ssh = FakeSsh(not_ready_for=10)
    monkeypatch.setattr(runner_module, "_ssh", ssh)

    def dying_spawn(name: str, command: list[str], log: Path, **kwargs) -> runner_module._Proc:
        proc = spawn(name, command, log, **kwargs)
        if name.startswith("env-"):
            proc.popen.returncode = 1
        return proc

    monkeypatch.setattr(runner_module, "_spawn", dying_spawn)

    result = leased(config, store).run(job, "run1")

    assert result.status is RunStatus.COMPLETED
    assert "forward-robosuite-0" not in spawn.commands
    assert any(cmd.startswith("docker rm -f") for _, cmd in ssh.calls)


def test_a_remote_container_and_its_directory_carry_its_gpu(remote_setup) -> None:
    """The leftover cleanup goes by GPU: another GPU of the host may be running
    another evaluation."""
    config, store, job, monkeypatch, spawn = remote_setup
    config.simulators["robosuite"].remote.gpu = "1"
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())

    leased(config, store).run(job, "run1")

    assert "--label capx.gpu=1 --gpus device=1" in spawn.commands["env-robosuite-0"][-1]
    ((_, _, remote_dir),) = spawn.copies
    assert remote_dir.startswith("/tmp/capx-remote/capx-") and remote_dir.endswith("-gpu-1")


def test_under_the_lease_every_capx_container_on_the_gpu_goes_before_the_gateway_starts(
    remote_setup,
) -> None:
    """The lease holder is the only evaluation on that GPU, so every capx container
    there is a killed worker's, of whichever queue. Clearing them while the submission
    runs would cost the team its time."""
    config, store, job, monkeypatch, spawn = remote_setup
    ssh = FakeSsh()
    gateway_started: list[bool] = []

    def recording_ssh(host: str, remote_command: str, timeout_s: float = 60.0) -> int:
        gateway_started.append("gateway" in spawn.commands)
        return ssh(host, remote_command, timeout_s)

    monkeypatch.setattr(runner_module, "_ssh", recording_ssh)
    runner = EvalRunner(config, store)

    assert runner.take_remote_gpus() is None
    runner.run(job, "run1")

    host, sweep = ssh.calls[0]
    assert host == "simbox" and not gateway_started[0]
    assert "docker ps -aq --filter label=capx.gpu=0)" in sweep
    assert sweep.endswith("rm -rf /tmp/capx-remote/capx-*-gpu-0")


def test_a_run_without_the_lease_is_refused(remote_setup) -> None:
    """Another worker may hold it, and the containers there may be its live run."""
    config, store, job, monkeypatch, spawn = remote_setup
    ssh = FakeSsh()
    monkeypatch.setattr(runner_module, "_ssh", ssh)

    with pytest.raises(RuntimeError, match="take_remote_gpus"):
        EvalRunner(config, store).run(job, "run1")

    assert ssh.calls == [] and spawn.commands == {}


def test_a_lease_lost_before_the_subset_stops_it_before_the_gateway(remote_setup) -> None:
    config, store, job, monkeypatch, spawn = remote_setup
    ssh = FakeSsh()
    monkeypatch.setattr(runner_module, "_ssh", ssh)
    runner = leased(config, store)
    (lease,) = runner._leases.values()
    lease.lose()

    with pytest.raises(LeaseLost, match="lost the lease on simbox GPU 0"):
        runner.run(job, "run1")

    assert spawn.commands == {}
    assert not any("ids=$(" in cmd for _, cmd in ssh.calls)


def test_a_lease_lost_mid_run_stops_it_and_starts_no_backend(remote_setup) -> None:
    """Another worker can take the GPU and clear this run's containers: what the run
    scores from then on says nothing about the submission."""
    config, store, job, monkeypatch, _ = remote_setup
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())
    runner = leased(config, store)
    (lease,) = runner._leases.values()
    spawn = FakeSpawn(runs_on=True)  # the submission runs until it is stopped

    def losing_spawn(name: str, command: list[str], log: Path, **kwargs) -> runner_module._Proc:
        proc = spawn(name, command, log, **kwargs)
        if name == "submission":  # which asks for its backends as it starts
            lease.lose()
        return proc

    monkeypatch.setattr(runner_module, "_spawn", losing_spawn)
    started = time.monotonic()

    with pytest.raises(LeaseLost):
        runner.run(job, "run1")

    assert time.monotonic() - started < PACK_LIMITS["trial_time_s"] / 2
    assert "env-robosuite-0" not in spawn.commands
    assert spawn.procs["submission"].popen.signals == [signal.SIGTERM]


def test_a_lease_lost_after_the_last_check_still_stops_the_run(remote_setup) -> None:
    config, store, job, monkeypatch, spawn = remote_setup
    runner = leased(config, store)
    (lease,) = runner._leases.values()
    ssh = FakeSsh()

    def losing_at_teardown(host: str, remote_command: str, timeout_s: float = 60.0) -> int:
        if remote_command.startswith("rm -rf /tmp/capx-remote/"):
            lease.lose()
        return ssh(host, remote_command, timeout_s)

    monkeypatch.setattr(runner_module, "_ssh", losing_at_teardown)

    with pytest.raises(LeaseLost):
        runner.run(job, "run1")


def test_a_cleanup_the_host_refuses_stops_the_run_before_the_gateway(remote_setup) -> None:
    config, store, job, monkeypatch, spawn = remote_setup

    def fenced_ssh(host: str, remote_command: str, timeout_s: float = 60.0) -> int:
        return runner_module.LEASE_FENCED if "ids=$(" in remote_command else 0

    monkeypatch.setattr(runner_module, "_ssh", fenced_ssh)

    with pytest.raises(LeaseLost, match="taken over"):
        leased(config, store).run(job, "run1")

    assert "gateway" not in spawn.commands


def test_a_backend_start_the_host_refuses_stops_the_run(remote_setup) -> None:
    config, store, job, monkeypatch, spawn = remote_setup
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())

    def fenced_spawn(name: str, command: list[str], log: Path, **kwargs) -> runner_module._Proc:
        proc = spawn(name, command, log, **kwargs)
        if name.startswith("env-"):
            assert command[-1].startswith("fence; ")
            proc.popen.returncode = runner_module.LEASE_FENCED
        return proc

    monkeypatch.setattr(runner_module, "_spawn", fenced_spawn)

    with pytest.raises(LeaseLost):
        leased(config, store).run(job, "run1")

    assert "forward-robosuite-0" not in spawn.commands


def test_a_run_whose_lease_the_host_no_longer_has_is_not_counted(remote_setup) -> None:
    config, store, job, monkeypatch, spawn = remote_setup
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())
    runner = leased(config, store)
    (lease,) = runner._leases.values()
    lease.confirmation = "simbox GPU 0 has been taken over since lease 1"

    with pytest.raises(LeaseLost, match="taken over since lease 1"):
        runner.run(job, "run1")

    assert lease.held()  # nothing local had noticed


def test_a_lease_that_ends_while_the_host_is_asked_still_stops_the_run(remote_setup) -> None:
    """The host can still answer for a lease whose ssh has just ended, or for a later
    lease while an earlier one ends."""
    config, store, job, monkeypatch, spawn = remote_setup
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())
    runner = leased(config, store)
    (lease,) = runner._leases.values()

    def confirm_while_ending() -> str | None:
        lease.lose()
        return None

    lease.confirm = confirm_while_ending

    with pytest.raises(LeaseLost, match="lost the lease on simbox GPU 0"):
        runner.run(job, "run1")


@pytest.mark.parametrize(
    "answer", [None, "simbox GPU 0 has been taken over since lease 1"], ids=["kept", "lost"]
)
def test_a_stop_while_the_host_is_asked_is_a_stop(remote_setup, answer) -> None:
    """Neither a result nor a counted loss: the job goes back as it was."""
    config, store, job, monkeypatch, spawn = remote_setup
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())
    runner = leased(config, store)
    (lease,) = runner._leases.values()

    def confirm_while_stopping() -> str | None:
        runner.interrupt()
        return answer

    lease.confirm = confirm_while_stopping

    with pytest.raises(Interrupted) as stopped:
        runner.run(job, "run1")

    assert not isinstance(stopped.value, LeaseLost)


def test_no_remote_backend_starts_without_a_live_lease(remote_setup, tmp_path) -> None:
    config, store, job, monkeypatch, spawn = remote_setup
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())
    key = ("simbox", "/tmp/capx-remote", "0")
    lease = FakeLease(key, lambda why: None)
    lease.lost = True
    root = tmp_path / "sockets"
    (root / "be").mkdir(parents=True)
    layout = runner_module._Layout(tmp_path, [], root, tmp_path)
    backends = runner_module._Backends(config, layout, ["robosuite"], {key: lease})

    with pytest.raises(LeaseLost, match="does not hold the lease on simbox GPU 0"):
        backends._start_remote("robosuite", 0, [])
    backends.close()

    assert spawn.commands == {}


def test_leftovers_that_cannot_be_removed_fail_the_subset_before_it_starts(remote_setup) -> None:
    config, store, job, monkeypatch, spawn = remote_setup

    def failing_ssh(host: str, remote_command: str, timeout_s: float = 60.0) -> int:
        return 1 if "ids=$(" in remote_command else 0

    monkeypatch.setattr(runner_module, "_ssh", failing_ssh)
    runner = EvalRunner(config, store)
    runner.take_remote_gpus()

    result = runner.run(job, "run1")

    assert result.status is RunStatus.FAILED
    assert result.error_code is ErrorCode.INTERNAL_ERROR
    assert "gateway" not in spawn.commands


@pytest.mark.parametrize(("over", "status"), [(0, RunStatus.COMPLETED), (1, RunStatus.FAILED)])
def test_remote_socket_paths_are_checked_for_the_last_lane_before_the_run(
    remote_setup, over, status
) -> None:
    """The kernel refuses a socket path over 107 bytes. Found only when the lane
    starts, it would void that lane's episodes while the job clock runs."""
    config, store, job, monkeypatch, spawn = remote_setup
    remote = config.simulators["robosuite"].remote
    config.simulators["robosuite"].max_lanes = 11
    # <socket_dir>/capx-<8 random>-gpu-0/robosuite-10.sock
    fixed = len("/capx-12345678-gpu-0") + len("/robosuite-10.sock")
    remote.socket_dir = "/" + "d" * (107 - fixed - 1 + over)
    monkeypatch.setattr(runner_module, "_ssh", FakeSsh())

    result = leased(config, store).run(job, "run1")

    assert result.status is status
    if status is RunStatus.FAILED:
        assert result.error_code is ErrorCode.INTERNAL_ERROR
        assert "gateway" not in spawn.commands
        assert "robosuite-10.sock" in result.subsets["public"].error


@pytest.fixture
def local_host(tmp_path, monkeypatch) -> Path:
    """The remote host, played by a local shell: ssh commands run in ``sh -c``, and a
    stand-in ``docker`` logs what it is asked to the returned file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {docker_log}\n'
        'case "$*" in\n'
        "  'ps -aq --filter label=capx.gpu=0') printf 'mine1\\nmine2\\n' ;;\n"
        "  'ps -aq --filter label=capx.gpu=1') echo other ;;\n"
        "esac\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(runner_module, "_ssh_argv", lambda host, command: ["sh", "-c", command])
    return docker_log


@pytest.fixture
def local_leases(local_host, tmp_path):
    """Leases taken with the real command on the local host, in ``tmp_path/locks``.
    ``lease.lost`` lists what each was told when it ended before its release."""
    leases: list[runner_module.RemoteGpuLease] = []

    def lease(gpu: str = "0") -> runner_module.RemoteGpuLease:
        lost: list[str] = []
        leases.append(runner_module.RemoteGpuLease(("simbox", str(tmp_path / "locks"), gpu), lost.append))
        leases[-1].lost = lost
        return leases[-1]

    yield lease
    for taken in leases:
        taken.release()


def test_the_cleanup_leaves_other_gpus_and_other_people_alone(local_host, local_leases, tmp_path) -> None:
    sockets = tmp_path / "locks"
    for name in ("capx-aaaa-gpu-0", "capx-bbbb-gpu-1", "capx-cccc-gpu-10", "someone-else"):
        (sockets / name).mkdir(parents=True)
    lease = local_leases()
    assert lease.take() is None

    runner_module._remove_leftovers(lease)

    assert local_host.read_text().splitlines() == [
        "ps -aq --filter label=capx.gpu=0",
        "rm -f mine1 mine2",
    ]
    assert sorted(path.name for path in sockets.iterdir()) == [
        "capx-bbbb-gpu-1",
        "capx-cccc-gpu-10",
        "capx-gpu-0.gen",
        "capx-gpu-0.lock",
        "capx-gpu-0.op",
        "someone-else",
    ]


def test_each_lease_on_a_gpu_counts_its_generation_up(local_leases) -> None:
    for generation in (1, 2, 3):
        lease = local_leases()
        assert lease.take() is None and lease.generation == generation
        lease.release()


def test_a_command_of_a_lease_taken_over_since_is_refused_on_the_host(
    local_host, local_leases, setup, tmp_path, monkeypatch
) -> None:
    """A's cleanup or backend start can stall on its way to the host while A loses the
    lease and B takes it. The host refuses it then, so B's containers stay."""
    config, _, _, _ = setup
    real_sleep = time.sleep
    monkeypatch.setattr(runner_module.time, "sleep", lambda s: real_sleep(min(s, 0.05)))
    config.simulators = {
        "robosuite": SimulatorConfig(
            remote=RemoteBackendConfig(host="simbox", socket_dir=str(tmp_path / "locks")),
            max_lanes=1,
        )
    }
    root = tmp_path / "sockets"
    (root / "be").mkdir(parents=True)
    layout = runner_module._Layout(tmp_path, [], root, tmp_path)
    key = ("simbox", str(tmp_path / "locks"), "0")
    first = local_leases()
    assert first.take() is None
    first.release()
    second = local_leases()
    assert second.take() is None
    first.held = lambda: True  # A's own look at its lease came before B took the GPU

    with pytest.raises(LeaseLost, match="taken over since lease 1"):
        runner_module._remove_leftovers(first)
    stale = runner_module._Backends(config, layout, ["robosuite"], {key: first})
    with pytest.raises(LeaseLost, match="taken over since lease 1"):
        stale._start_remote("robosuite", 0, [])
    stale.close()

    # Nothing but the stop of A's own container, by a name no other run's has.
    assert set(local_host.read_text().splitlines()) == {"rm -f sockets-robosuite-0"}
    current = runner_module._Backends(config, layout, ["robosuite"], {key: second})
    with pytest.raises(SubsetFailed) as failed:
        current._start_remote("robosuite", 0, [])  # the stand-in docker makes no socket
    current.close()
    assert not isinstance(failed.value, LeaseLost)
    asked = local_host.read_text().splitlines()
    assert any(line.startswith("create --rm --name sockets-robosuite-0 ") for line in asked)
    assert "start -a sockets-robosuite-0" in asked  # outside the op lock, once created


def test_a_remote_gpu_has_one_lease_holder_at_a_time(local_leases) -> None:
    first, second = local_leases(), local_leases()

    assert first.take() is None and first.held()
    assert second.take() == "simbox GPU 0 is in use by another evaluation"
    first.release()
    assert second.take() is None


def test_the_gpus_of_one_host_are_leased_apart(local_leases) -> None:
    assert local_leases("0").take() is None
    assert local_leases("1").take() is None


HOLDER = """
import subprocess, sys, time
lease = subprocess.Popen(["sh", "-c", sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
sys.stdout.buffer.write(lease.stdout.readline())
sys.stdout.flush()
time.sleep(600)
"""


def test_a_killed_worker_s_lease_goes_with_it(local_leases, tmp_path) -> None:
    """Its end of the pipe closes, the remote side reads EOF and drops the lock:
    nothing is left to clean up by hand."""
    command = runner_module._lease_command(str(tmp_path / "locks"), "0")
    holder = subprocess.Popen([sys.executable, "-I", "-c", HOLDER, command], stdout=subprocess.PIPE)
    try:
        assert holder.stdout.readline().startswith(runner_module.LEASE_HELD.encode())
        lease = local_leases()
        assert lease.take() is not None
    finally:
        holder.kill()
        holder.wait()
    _eventually(lambda: lease.take() is None)


def test_a_lease_whose_ssh_dies_mid_job_says_so(local_leases) -> None:
    lease = local_leases()
    assert lease.take() is None

    lease._proc.kill()

    _eventually(lambda: lease.lost == ["lost the lease on simbox GPU 0 (ssh exit -9)"])
    assert not lease.held()


def test_a_released_lease_is_not_reported_lost(local_leases) -> None:
    lease = local_leases()
    assert lease.take() is None

    lease.release()

    assert lease.lost == [] and not lease.held()


@pytest.mark.parametrize(
    ("argv", "busy"),
    [
        (["sh", "-c", "exit 255"], "could not take the lease on simbox GPU 0 (exit 255)"),
        (["sleep", "60"], "simbox did not answer within 0s for GPU 0's lease"),
    ],
)
def test_a_host_that_does_not_answer_gives_no_lease(local_leases, monkeypatch, argv, busy) -> None:
    monkeypatch.setattr(runner_module, "_ssh_argv", lambda host, command: argv)
    lease = local_leases()

    assert lease.take(timeout_s=0.2) == busy
    assert not lease.held()


@pytest.mark.parametrize(
    ("remote", "generation"),
    [
        # One write, so one read: the marker comes in the same chunk as the banner.
        ("printf 'welcome\\ncapx-lease-held 7\\n'; exec cat >/dev/null", 7),
        # A banner with no newline, then the real lock.
        ("printf welcome; {command}", 1),
    ],
    ids=["same-chunk", "unterminated-banner"],
)
def test_a_lease_reads_past_what_the_remote_login_prints(
    local_leases, monkeypatch, remote, generation
) -> None:
    monkeypatch.setattr(
        runner_module,
        "_ssh_argv",
        lambda host, command: ["sh", "-c", remote.format(command=command)],
    )
    lease = local_leases()
    assert lease.take(timeout_s=5.0) is None and lease.generation == generation


def test_a_banner_that_never_ends_does_not_hold_up_the_take(local_leases, monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module, "_ssh_argv", lambda host, command: ["sh", "-c", "printf welcome; exec sleep 60"]
    )
    started = time.monotonic()

    assert "did not answer" in local_leases().take(timeout_s=0.3)
    assert time.monotonic() - started < 5.0


def test_a_new_lease_waits_for_a_cleanup_already_past_its_fence(
    local_host, local_leases, tmp_path
) -> None:
    """A passes its fence, stalls in docker, and loses the lease meanwhile. B's lease
    counts the generation up only once A's cleanup is through, so nothing A's cleanup
    does reaches what B starts, and A's later commands are refused."""
    stalled = tmp_path / "stalled"
    (tmp_path / "bin" / "docker").write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {local_host}\n'
        'case "$*" in\n'
        f"  'ps -aq --filter label=capx.gpu=0') touch {stalled}; sleep 1; echo mine1 ;;\n"
        "esac\n"
    )
    first = local_leases()
    assert first.take() is None
    cleanup = threading.Thread(target=runner_module._remove_leftovers, args=(first,))
    cleanup.start()
    _eventually(stalled.exists)
    first.release()
    second = local_leases()

    assert second.take() is None and second.generation == 2
    done_by_then = local_host.read_text().splitlines()
    cleanup.join()
    with pytest.raises(LeaseLost):
        runner_module._remove_leftovers(first)

    assert done_by_then == ["ps -aq --filter label=capx.gpu=0", "rm -f mine1"]
    assert local_host.read_text().splitlines() == done_by_then


def test_a_result_needs_the_host_to_still_have_the_lease(local_leases, tmp_path) -> None:
    """In a network partition the host can drop the lock, and another worker take it,
    while this worker's ssh client still waits out its keepalive: locally the lease
    still looks held."""
    lease = local_leases()
    assert lease.take() is None and lease.confirm() is None

    (tmp_path / "locks" / "capx-gpu-0.gen").write_text("2\n")

    assert lease.held()
    assert lease.confirm() == "simbox GPU 0 has been taken over since lease 1"


def test_a_result_is_not_counted_when_the_host_cannot_say(local_leases, monkeypatch) -> None:
    lease = local_leases()
    assert lease.take() is None

    monkeypatch.setattr(runner_module, "_ssh", lambda host, command, timeout_s=60.0: 255)

    assert lease.confirm() == "could not confirm the lease on simbox GPU 0 (exit 255)"


def test_two_spellings_of_one_socket_dir_are_one_lease(local_host, tmp_path) -> None:
    """Two leases on one lock file: the worker would wait on itself for ever."""
    locks = str(tmp_path / "locks")
    config = QueueConfig(
        root=str(tmp_path / "root"),
        master_seed=1,
        simulators={
            sim: {"remote": {"host": "simbox", "socket_dir": socket_dir}, "max_lanes": 1}
            for sim, socket_dir in (("a", locks), ("b", f"{locks}/"))
        },
        subsets=[{"name": "s", "taskpack": "taskpacks/original"}],
        slurm={"workers": 1},
    )
    runner = EvalRunner(config, EvalStore(config.root))

    assert runner.take_remote_gpus() is None
    runner.release_remote_gpus()


def test_a_busy_gpu_releases_the_leases_already_taken(local_leases, tmp_path) -> None:
    """Holding GPU 0 while waiting for GPU 1 would keep GPU 0's own queue waiting too."""
    locks = str(tmp_path / "locks")
    config = QueueConfig(
        root=str(tmp_path / "root"),
        master_seed=1,
        simulators={
            sim: {"remote": {"host": "simbox", "socket_dir": locks, "gpu": gpu}, "max_lanes": 1}
            for sim, gpu in (("a", "0"), ("b", "1"))
        },
        subsets=[{"name": "s", "taskpack": "taskpacks/original"}],
        slurm={"workers": 1},
    )
    runner = EvalRunner(config, EvalStore(config.root))
    assert local_leases("1").take() is None

    assert runner.take_remote_gpus() == "simbox GPU 1 is in use by another evaluation"
    assert local_leases("0").take() is None


def test_a_simulator_runs_one_way() -> None:
    with pytest.raises(ValueError, match="exactly one of image or remote"):
        SimulatorConfig(image="x.sif", remote={"host": "simbox"}, max_lanes=1)
    with pytest.raises(ValueError, match="exactly one of image or remote"):
        SimulatorConfig(max_lanes=1)


def test_the_old_run_section_is_refused() -> None:
    with pytest.raises(ValueError, match="run"):
        QueueConfig.model_validate({"run": {"job_timeout_s": 60.0}})


def _eventually(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        time.sleep(0.01)


def test_a_local_backend_that_never_creates_its_socket_is_retired(setup, tmp_path) -> None:
    """Left running, its lane would be skipped for the rest of the run: the name stays
    requested, so the gateway's next ask for it would never start a fresh one."""
    config, _, _, monkeypatch = setup
    config.limits.backend_startup_s = 0.05
    root = tmp_path / "sockets"
    (root / "be").mkdir(parents=True)
    layout = runner_module._Layout(tmp_path, [], root, tmp_path)
    backends = runner_module._Backends(config, layout, ["robosuite"], {})
    starts: list[FakePopen] = []

    def start_local(simulator, lane):
        starts.append(FakePopen())  # alive, and never binds its socket
        return runner_module._Proc(name=f"env-{simulator}-{lane}", popen=starts[-1], log=tmp_path / "log")

    monkeypatch.setattr(backends, "_start_local", start_local)
    want = root / "be" / "robosuite-0.sock.want"
    try:
        want.touch()
        backends.poll()
        _eventually(lambda: not want.exists() and not backends._procs)
        assert signal.SIGTERM in starts[0].signals

        want.touch()
        backends.poll()
        _eventually(lambda: len(starts) == 2)
    finally:
        backends.close()


def test_closing_does_not_wait_out_a_backend_start(setup, tmp_path) -> None:
    """A start waits up to backend_startup_s for the socket; the end of the subset
    must not wait with it."""
    config, _, _, monkeypatch = setup
    config.limits.backend_startup_s = 600.0
    root = tmp_path / "sockets"
    (root / "be").mkdir(parents=True)
    layout = runner_module._Layout(tmp_path, [], root, tmp_path)
    backends = runner_module._Backends(config, layout, ["robosuite"], {})
    starts: list[FakePopen] = []

    def start_local(simulator, lane):
        starts.append(FakePopen())  # alive, and never binds its socket
        return runner_module._Proc(name=f"env-{simulator}-{lane}", popen=starts[-1], log=tmp_path / "log")

    monkeypatch.setattr(backends, "_start_local", start_local)
    (root / "be" / "robosuite-0.sock.want").touch()
    backends.poll()
    _eventually(lambda: len(backends._procs) == 1)

    began = time.monotonic()
    backends.close()
    assert time.monotonic() - began < 10
    assert signal.SIGTERM in starts[0].signals


def test_a_start_that_fails_late_leaves_its_replacement_alone(setup, tmp_path) -> None:
    """The first start's process died and was retired, and the gateway asked again,
    before that start gave up waiting. Its clean-up must not reach the second one."""
    config, _, _, monkeypatch = setup
    config.simulators["robosuite"].max_lanes = 2  # a thread for the retire beside the stuck start
    root = tmp_path / "sockets"
    (root / "be").mkdir(parents=True)
    layout = runner_module._Layout(tmp_path, [], root, tmp_path)
    backends = runner_module._Backends(config, layout, ["robosuite"], {})
    first, second = FakePopen(), FakePopen()
    popens = iter([first, second])
    first_gives_up = threading.Event()

    def start_local(simulator, lane):
        return runner_module._Proc(name=f"env-{simulator}-{lane}", popen=next(popens), log=tmp_path / "log")

    def wait_until(ready, proc, *args, **kwargs):
        if proc.popen is first:
            first_gives_up.wait(5)
            raise SubsetFailed("gave up")

    monkeypatch.setattr(backends, "_start_local", start_local)
    monkeypatch.setattr(runner_module, "_wait_until", wait_until)
    want = root / "be" / "robosuite-0.sock.want"
    try:
        want.touch()
        backends.poll()
        _eventually(lambda: len(backends._procs) == 1)
        first.returncode = 1  # it died; the next poll retires it
        backends.poll()
        _eventually(lambda: not want.exists())
        want.touch()  # the gateway asks again
        backends.poll()
        _eventually(lambda: second in [proc.popen for proc in backends._procs])

        first_gives_up.set()
        _eventually(lambda: first not in [proc.popen for proc in backends._procs])
        assert second.signals == []
        assert [proc.popen for proc in backends._running["robosuite-0"]] == [second]
        assert want.exists() and "robosuite-0" in backends._requested
    finally:
        first_gives_up.set()
        backends.close()


def test_a_backend_that_exits_starts_again_when_next_asked_for(setup, tmp_path) -> None:
    """Its socket file outlives it; left there, the gateway would keep connecting to
    a dead socket and every episode on that lane would be voided."""
    config, _, _, monkeypatch = setup
    root = tmp_path / "sockets"
    (root / "be").mkdir(parents=True)
    layout = runner_module._Layout(tmp_path, [], root, tmp_path)
    backends = runner_module._Backends(config, layout, ["robosuite"], {})
    procs = backends._procs
    starts: list[FakePopen] = []

    def start_local(simulator, lane):
        popen = FakePopen()
        starts.append(popen)
        (root / "be" / f"{simulator}-{lane}.sock").touch()
        return runner_module._Proc(name=f"env-{simulator}-{lane}", popen=popen, log=tmp_path / "log")

    monkeypatch.setattr(backends, "_start_local", start_local)
    want = root / "be" / "robosuite-0.sock.want"
    try:
        want.touch()
        backends.poll()
        _eventually(lambda: len(procs) == 1)

        starts[0].returncode = 1  # the backend died
        backends.poll()
        _eventually(lambda: not want.exists() and not procs)
        assert not (root / "be" / "robosuite-0.sock").exists()

        want.touch()  # the gateway asks again on its next episode
        backends.poll()
        _eventually(lambda: len(starts) == 2 and len(procs) == 1)
    finally:
        backends.close()


def test_only_the_gateway_mounts_the_recordings(setup) -> None:
    config, store, job, monkeypatch = setup
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    gateway = spawn.commands["gateway"]
    videos = Path(_env_value(gateway, "CAPX_RECORDINGS_DIR"))
    assert videos == store.recordings / "run1" / "public"
    assert f"{videos}:{videos}" in _binds(gateway)
    assert videos.stat().st_mode & 0o077 == 0
    for name, command in spawn.commands.items():
        if name == "gateway":
            continue
        for bind in _binds(command):
            source = Path(bind.split(":")[0])
            assert not (source == store.recordings or store.recordings in source.parents), name
            assert not (source in store.recordings.parents), (name, source)


def test_a_run_without_recordings_mounts_none(setup) -> None:
    config, store, job, monkeypatch = setup
    config.recordings.enabled = False
    spawn = FakeSpawn()
    monkeypatch.setattr(runner_module, "_spawn", spawn)
    EvalRunner(config, store).run(job, "run1")

    assert "CAPX_RECORDINGS_DIR" not in _env(spawn.commands["gateway"])
    assert not (store.recordings / "run1").exists()
