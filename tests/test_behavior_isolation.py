from pathlib import Path
import os
import tempfile

import pytest

from capx.gateway.behavior_isolation import IsolatedBehaviorFactory
from capx.protocol.taskpack import TaskSpec


@pytest.fixture
def isolated_factory(tmp_path, monkeypatch):
    script = tmp_path / "cpu_worker.py"
    script.write_text('''import os
from capx.gateway.env_backend import EnvBackendServer
from capx.gateway.low_level import StepOutcome
class Env:
    embodiment = 'r1pro'
    sim_time = 0.0
    def reset(self, *, seed): pass
    def apply_raw_action(self, action): return StepOutcome()
    def get_proprio(self): return {'pid': os.getpid()}
    def get_observation(self): return self.get_proprio()
    def compute_reward(self): return 0.0
    def close(self): pass
class Factory:
    def build(self, spec): return Env()
EnvBackendServer(Factory(), os.environ['CAPX_BACKEND_UNIX'], main_thread=True).serve_forever()
''')
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    with tempfile.TemporaryDirectory(prefix="bwv1-") as root:
        monkeypatch.setenv("CAPX_WORKER_ROOT", root)
        yield IsolatedBehaviorFactory(script)


def spec():
    return TaskSpec(task_id="fake", simulator="behavior", embodiment="r1pro",
                    instruction="CPU isolation regression", env_target="fake", n_trials=1,
                    trial_steplimit=20, seed_policy={"mode": "fixed", "seeds": [1]})


def test_same_lane_process_uses_fresh_workers_and_fully_releases_them(isolated_factory):
    parent = os.getpid()
    first = isolated_factory.build(spec())
    first.reset(seed=1)
    first_pid = first.get_proprio()["pid"]
    first.close()
    first.close()
    assert first._process.poll() is not None
    second = isolated_factory.build(spec())
    try:
        second_pid = second.get_proprio()["pid"]
        assert parent == os.getpid()
        assert first_pid != second_pid != parent
        assert second.compute_reward() == 0.0
    finally:
        second.close()
    assert second._process.poll() is not None


def test_a_worker_crash_does_not_poison_the_next_environment(isolated_factory):
    first = isolated_factory.build(spec())
    first._process.kill()
    first._process.wait(timeout=10)
    with pytest.raises((OSError, EOFError, ConnectionError)):
        first.get_observation()
    first.close()
    second = isolated_factory.build(spec())
    try:
        second.reset(seed=1)
        assert second.get_observation()["pid"] == second._process.pid
    finally:
        second.close()
