"""Organizer-only episode videos: which episodes are filmed, how frames get from the
backend to an MP4, and that nothing about it can break an episode."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import pytest
import tyro

from capx.cli.evalq import SuperviseArgs, WorkerArgs
from capx.evalqueue import slurm
from capx.evalqueue.config import QueueConfig
from capx.evalqueue.recordings import DAY_S, prune_recordings, recordings_dir
from capx.evalqueue.store import EvalStore
from capx.gateway import EnforcementMode, Gateway
from capx.gateway.env_backend import EnvBackendServer, RemoteEnvFactory
from capx.gateway.low_level import SimAdapter, encode_jpeg
from capx.gateway.recording import Recorder, choose_episodes
from capx.protocol import EnvClient, InProcessTransport
from capx.protocol.actions import FrankaRawAction
from test_gateway import VALID_JOINTS, FakeEnv, FakeFactory, make_tasks


def act() -> FrankaRawAction:
    return FrankaRawAction(joints=VALID_JOINTS, gripper=1.0)


class FakeAdapter(SimAdapter):
    embodiment = "franka"

    def __init__(self, *, fail_render: bool = False) -> None:
        self.now = 0.0
        self.renders = 0
        self._fail_render = fail_render

    @property
    def sim_time(self) -> float:
        return self.now

    def _render_frame(self) -> np.ndarray:
        if self._fail_render:
            raise RuntimeError("no GL context")
        self.renders += 1
        return np.full((16, 16, 3), self.renders, dtype=np.uint8)


def test_frames_are_captured_on_a_sim_time_grid() -> None:
    adapter = FakeAdapter()
    adapter._capture_if_due(0.0)
    assert adapter.renders == 0  # nothing before start_recording

    adapter.start_recording(10.0)
    for now in (0.0, 0.05, 0.1, 0.12, 0.25):
        adapter._capture_if_due(now)
    frames = adapter.stop_recording()

    assert len(frames) == 3
    assert all(frame[:2] == b"\xff\xd8" for frame in frames)  # JPEG
    assert adapter.stop_recording() == []


def test_a_failed_render_ends_the_recording_not_the_episode() -> None:
    adapter = FakeAdapter(fail_render=True)
    adapter.start_recording(10.0)
    adapter._capture_if_due(0.0)
    adapter._capture_if_due(1.0)
    assert adapter.stop_recording() == []


def test_episodes_are_chosen_per_simulator() -> None:
    specs = [
        *make_tasks(task_id="a", n_trials=10).values(),
        *make_tasks(task_id="b", n_trials=10).values(),
        *make_tasks(task_id="c", simulator="libero", n_trials=3).values(),
    ]
    tasks = {spec.task_id: spec for spec in specs}
    chosen = choose_episodes(tasks, 10, random.Random(0))

    assert sum(task in ("a", "b") for task, _ in chosen) == 10
    assert {pair for pair in chosen if pair[0] == "c"} == {("c", 0), ("c", 1), ("c", 2)}
    assert all(0 <= trial < tasks[task].n_trials for task, trial in chosen)


class RecordingEnv(FakeEnv):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.recording = False

    def start_recording(self, fps: float) -> None:
        self.recording = True

    def stop_recording(self) -> list[bytes]:
        self.recording = False
        return [encode_jpeg(np.full((32, 48, 3), i * 20, dtype=np.uint8)) for i in range(5)]


class RecordingFactory(FakeFactory):
    def build(self, spec) -> RecordingEnv:
        env = RecordingEnv(**self._kwargs)
        self.built.append(env)
        return env


def _remote_gateway(
    factory: FakeFactory, recorder: Recorder
) -> tuple[EnvClient, Gateway, EnvBackendServer]:
    server = EnvBackendServer(factory).start_background()
    gateway = Gateway(
        make_tasks(),
        RemoteEnvFactory({"robosuite": [server.address]}),
        EnforcementMode.OFFICIAL,
        recorder=recorder,
    )
    return EnvClient(InProcessTransport(gateway.acquire())), gateway, server


def test_a_chosen_episode_is_written_as_mp4_through_the_backend(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, {("t", 0)})
    client, gateway, server = _remote_gateway(RecordingFactory(solve_at_cycle=1), recorder)
    try:
        first = client.reset("t")
        client.step(first.episode_id, act())
        second = client.reset("t")  # trial 1: not chosen
        client.step(second.episode_id, act())
        gateway.close()
        assert recorder.close(30.0)
    finally:
        server.close()

    videos = sorted(p.name for p in (tmp_path / "robosuite").iterdir())
    assert videos == ["t__trial0__success.mp4"]
    data = (tmp_path / "robosuite" / videos[0]).read_bytes()
    assert data[4:8] == b"ftyp"


def test_a_backend_that_cannot_record_still_runs_the_episode(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, {("t", 0), ("t", 1)})
    client, gateway, server = _remote_gateway(FakeFactory(), recorder)
    try:
        reset = client.reset("t")
        client.step(reset.episode_id, act())
        gateway.close()
        assert recorder.close(30.0)
    finally:
        server.close()
    assert gateway.trials_used("t") == 1
    assert not any(tmp_path.iterdir())


def test_recordings_are_deleted_after_keep_days(tmp_path: Path) -> None:
    store = EvalStore(tmp_path / "root")
    store.ensure_layout()
    assert store.recordings.stat().st_mode & 0o077 == 0
    old = recordings_dir(store, "team__a__20261101T000000Z", "pack")
    (old / "v.mp4").write_bytes(b"x")
    new = recordings_dir(store, "team__b__20261112T000000Z", "pack")
    (new / "v.mp4").write_bytes(b"x")
    eight_days_ago = time.time() - 8 * DAY_S
    for path in (old / "v.mp4", old, old.parent):
        os.utime(path, (eight_days_ago, eight_days_ago))

    assert prune_recordings(store, 7.0) == ["team__a__20261101T000000Z"]
    assert not old.parent.exists()
    assert (new / "v.mp4").exists()


@pytest.mark.parametrize("args_type", [WorkerArgs, SuperviseArgs])
def test_no_recordings_flag(args_type: type) -> None:
    assert tyro.cli(args_type, args=["--no-recordings"]).no_recordings
    assert not tyro.cli(args_type, args=[]).no_recordings


def test_supervise_passes_no_recordings_to_the_worker(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "scripts" / "evalq" / "worker.sbatch"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n")
    commands: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = "123\n"
        stderr = ""

    def fake_run(command: list[str], **_kwargs: object) -> Done:
        commands.append(command)
        return Done()

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    config = QueueConfig(master_seed=1, slurm={"partition": "p"})
    slurm.submit_worker(config, tmp_path, recordings=False)
    slurm.submit_worker(config, tmp_path)
    assert "--no-recordings" in commands[0]
    assert "--no-recordings" not in commands[1]
