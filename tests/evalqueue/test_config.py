from __future__ import annotations

import pytest

from capx.evalqueue.config import QueueConfig, RemoteBackendConfig, SubsetConfig, load_queue_config


def test_a_config_file_sets_only_what_differs_from_the_defaults(tmp_path) -> None:
    path = tmp_path / "round.yaml"
    path.write_text("root: ~/rounds/r1\nmaster_seed: 7\nslurm:\n  workers: 4\n")
    config = load_queue_config(str(path))
    defaults = QueueConfig(master_seed=7)
    assert config.root == "~/rounds/r1"
    assert config.slurm.workers == 4
    assert config.slurm.mem == defaults.slurm.mem
    assert config.simulators == defaults.simulators


def test_an_official_round_needs_its_own_seed(tmp_path) -> None:
    """The seed derivation is public; the default would let a submission compute every
    episode's seed."""
    with pytest.raises(ValueError, match="master_seed is required"):
        load_queue_config()
    path = tmp_path / "round.yaml"
    path.write_text("root: ~/rounds/r1\n")
    with pytest.raises(ValueError, match="master_seed is required"):
        load_queue_config(str(path))
    assert QueueConfig(mode="dev").master_seed == 0
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        QueueConfig(master_seed=-1)


def test_the_evalq_cli_imports() -> None:
    """A dataclass field-ordering mistake once broke every `capx evalq` subcommand."""
    import capx.cli.evalq  # noqa: F401


def test_a_config_needs_a_subset_to_score_on() -> None:
    with pytest.raises(ValueError, match="subsets"):
        QueueConfig.model_validate({"subsets": [], "master_seed": 1})


def test_subset_names_are_unique() -> None:
    """Results and run directories are keyed by name: a second pack of the same name
    would overwrite the first one's result."""
    with pytest.raises(ValueError, match="subset names must be unique: a"):
        QueueConfig(
            master_seed=1,
            subsets=[
                SubsetConfig(name="a", taskpack="taskpacks/original"),
                SubsetConfig(name="a", taskpack="taskpacks/other"),
            ],
        )


REMOTE_SIMULATORS = {
    "robosuite": {"image": "env-robosuite.sif", "max_lanes": 32},
    "behavior": {"remote": {"host": "simbox"}, "max_lanes": 8},
}


def _with_subset_on(simulators: list[str], **settings) -> dict:
    subset = {"name": "s", "taskpack": "taskpacks/original", "simulators": simulators}
    return {"master_seed": 1, "simulators": REMOTE_SIMULATORS, "subsets": [subset], **settings}


def test_one_evaluation_at_a_time_uses_the_remote_gpu() -> None:
    """Every worker's backends land on the same remote GPU, and two evaluations' do not
    fit on it."""
    with pytest.raises(ValueError, match="slurm.workers must be 1 .*behavior.*not shared"):
        QueueConfig.model_validate(_with_subset_on(["robosuite", "behavior"]))
    config = QueueConfig.model_validate(
        _with_subset_on(["robosuite", "behavior"], slurm={"workers": 1})
    )
    assert config.slurm.workers == 1


def test_a_subset_without_a_simulator_list_may_use_the_remote_one() -> None:
    with pytest.raises(ValueError, match="slurm.workers must be 1"):
        QueueConfig.model_validate(_with_subset_on([]))


def test_a_remote_simulator_no_subset_uses_does_not_limit_the_workers() -> None:
    config = QueueConfig.model_validate(_with_subset_on(["robosuite"], slurm={"workers": 2}))
    assert config.slurm.workers == 2


def test_a_deadline_needs_a_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        QueueConfig(deadline="2026-12-13T12:00:00", master_seed=1)
    assert QueueConfig(deadline="2026-12-13T12:00:00+09:00", master_seed=1).deadline.utcoffset().total_seconds() == 9 * 3600


@pytest.mark.parametrize(
    "gpu", ["0,1", "all", "GPU-5f2a8c1e-0b6d-4c7a-9e21-3f0d5b7a9c44", "-1", "00", "01"]
)
def test_a_remote_simulator_names_one_gpu_by_index(gpu: str) -> None:
    """Evaluations take turns per GPU, on a lock named for it: a lease on "0,1" or "all"
    would not keep one off "0", and a UUID or "00" would be a second lock for "0"'s GPU."""
    with pytest.raises(ValueError, match="gpu"):
        RemoteBackendConfig(host="simbox", gpu=gpu)
    assert RemoteBackendConfig(host="simbox", gpu="1").gpu == "1"


def test_a_remote_socket_dir_has_one_spelling() -> None:
    """The GPU's lock file is found by it: "/tmp" and "/tmp/" would be two leases."""
    assert RemoteBackendConfig(host="simbox", socket_dir="/tmp/").socket_dir == "/tmp"
    assert RemoteBackendConfig(host="simbox", socket_dir="//tmp/./capx//").socket_dir == "/tmp/capx"
    with pytest.raises(ValueError, match="absolute"):
        RemoteBackendConfig(host="simbox", socket_dir="tmp")


@pytest.mark.parametrize("socket_dir", ["/tmp/link/..", "/tmp/../var/tmp", "/.."])
def test_a_remote_socket_dir_with_dot_dot_is_refused(socket_dir: str) -> None:
    """The host resolves ".." after symlinks: "/tmp/link/.." need not be "/tmp"."""
    with pytest.raises(ValueError, match=r"'\.\.'"):
        RemoteBackendConfig(host="simbox", socket_dir=socket_dir)
