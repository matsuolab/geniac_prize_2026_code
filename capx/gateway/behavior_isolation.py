"""Keep a lane's backend alive while each Isaac lifecycle owns a fresh process.

OmniGibson's process-global PhysX/USD/material caches can survive scene clear.
An exec-created worker makes closing an episode release those caches completely.
Only private Unix sockets are used; task specs and observation filtering are intact.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from capx.gateway.env_backend import BackendAddress, RemoteEnvFactory


class OwnedBehaviorEnv:
    def __init__(self, remote, process, log, directory):
        self._remote, self._process, self._log = remote, process, log
        self._directory = directory
        self.embodiment = remote.embodiment
        self._closed = False

    @property
    def sim_time(self):
        return self._remote.sim_time

    def reset(self, *, seed):
        return self._remote.reset(seed=seed)

    def apply_raw_action(self, action):
        return self._remote.apply_raw_action(action)

    def get_proprio(self):
        return self._remote.get_proprio()

    def get_observation(self):
        return self._remote.get_observation()

    def compute_reward(self):
        return self._remote.compute_reward()

    def start_recording(self, fps):
        return self._remote.start_recording(fps)

    def stop_recording(self):
        return self._remote.stop_recording()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._remote.close()
        finally:
            stop_owned_worker(self._process)
            self._log.close()
            (self._directory / "closed.json").write_text(json.dumps({
                "worker_pid": self._process.pid, "returncode": self._process.returncode,
                "worker_exited": True,
            }, indent=2) + "\n")


def stop_owned_worker(process):
    if process.poll() is None:
        # Popen has not reaped this PID, so it cannot be reused by another job.
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


class IsolatedBehaviorFactory:
    supports_parallel = False

    def __init__(self, worker_script=None):
        self._worker_script = Path(worker_script) if worker_script else Path(__file__).resolve().parents[2] / "docker/serve_env_backend.py"

    def build(self, spec):
        root = Path(os.environ.get("CAPX_WORKER_ROOT", "/work/w"))
        root.mkdir(parents=True, exist_ok=True)
        directory = root / uuid.uuid4().hex[:10]
        directory.mkdir(mode=0o700)
        sock = directory / "b.sock"
        if len(os.fsencode(sock)) > 107:
            raise ValueError("CAPX_WORKER_ROOT is too long for a private Unix socket")
        env = dict(os.environ, CAPX_BEHAVIOR_WORKER="1", CAPX_BACKEND_UNIX=str(sock))
        env.pop("CAPX_BACKEND_UNIX_OWNER", None)
        log = (directory / "worker.log").open("x")
        command = [sys.executable, "-u", "-X", "faulthandler", str(self._worker_script)]
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (directory / "process.json").write_text(json.dumps({
            "pid": process.pid, "command": command, "socket": str(sock),
            "task_id": spec.task_id, "spec": spec.model_dump(mode="json"),
        }, indent=2) + "\n")
        try:
            factory = RemoteEnvFactory({spec.simulator: [BackendAddress(str(sock))]},
                start_timeout_s=240, call_timeout_s=1800)
            remote = factory.build(spec)
        except BaseException:
            stop_owned_worker(process)
            log.close()
            raise
        return OwnedBehaviorEnv(remote, process, log, directory)
