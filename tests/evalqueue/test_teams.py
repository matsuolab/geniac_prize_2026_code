"""The team registry: one file per team, read out of git."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from capx.evalqueue.config import TeamsConfig
from capx.evalqueue.store import EvalStore
from capx.evalqueue.teams import TeamRegistry, parse_registry, registered_teams

REPO_ROOT = Path(__file__).resolve().parents[2]


def _file(team_id: str, org: str | None = None) -> str:
    return f"team_id: {team_id}\nrepo: {org or team_id}/capx-submission\n"


def test_a_registration_is_read_from_its_file() -> None:
    teams, problems = parse_registry({"alpha.yaml": _file("alpha")})
    assert [(t.team_id, t.repo) for t in teams] == [("alpha", "alpha/capx-submission")]
    assert problems == []


@pytest.mark.parametrize(
    ("name", "text", "problem"),
    [
        ("beta.yaml", _file("alpha"), "must be named alpha.yaml"),
        ("alpha.yaml", _file("alpha", "alpha-org"), "team_id must be the repo's org, alpha-org"),
        ("alpha.yaml", "team_id: [", "alpha.yaml"),
        ("alpha.yaml", "team_id: ../etc\nrepo: org/repo\n", "team_id"),
        ("alpha.yaml", "team_id: alpha\nrepo: no-slash\n", "must be a HuggingFace '<owner>/<name>'"),
        ("alpha.yaml", "team_id: alpha\nrepo: org/a/b\n", "must be a HuggingFace '<owner>/<name>'"),
        # The watched branch is always main, so a team cannot pick another.
        ("alpha.yaml", _file("alpha") + "revision: dev\n", "revision"),
    ],
)
def test_a_bad_registration_is_left_out(name: str, text: str, problem: str) -> None:
    teams, problems = parse_registry({name: text, "gamma.yaml": _file("gamma")})
    assert [t.team_id for t in teams] == ["gamma"]
    assert len(problems) == 1 and problem in problems[0]


def test_teams_whose_orgs_differ_only_in_case_are_all_left_out() -> None:
    teams, problems = parse_registry(
        {"alpha.yaml": _file("alpha"), "Alpha.yaml": _file("Alpha"), "gamma.yaml": _file("gamma")}
    )
    assert [t.team_id for t in teams] == ["gamma"]
    assert problems == ["org 'alpha' is named by more than one team: Alpha, alpha"]


def test_the_registrations_in_this_repository_are_valid() -> None:
    files = {path.name: path.read_text() for path in (REPO_ROOT / "teams" / "prelim").glob("*.yaml")}
    assert parse_registry(files)[1] == []


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
    )


def _commit(repo: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = repo / "teams" / "prelim" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "register")


@pytest.fixture
def registry(tmp_path: Path) -> tuple[Path, EvalStore, TeamRegistry]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, {"alpha.yaml": _file("alpha"), "README.md": "not a team"})
    store = EvalStore(tmp_path / "root")
    store.ensure_layout()
    return repo, store, TeamRegistry(TeamsConfig(repo=str(repo)), store)


def test_a_merged_registration_is_picked_up_on_the_next_refresh(registry) -> None:
    repo, store, teams = registry
    assert [t.team_id for t in teams.refresh()] == ["alpha"]
    _commit(repo, {"beta.yaml": _file("beta")})
    assert [t.team_id for t in teams.refresh()] == ["alpha", "beta"]
    assert [t.team_id for t in registered_teams(store)] == ["alpha", "beta"]


def test_a_registration_on_another_branch_does_not_count(registry) -> None:
    repo, _, teams = registry
    _git(repo, "checkout", "-q", "-b", "pr")
    _commit(repo, {"beta.yaml": _file("beta")})
    assert [t.team_id for t in teams.refresh()] == ["alpha"]


def test_an_unreachable_registry_keeps_the_last_list(registry, tmp_path: Path) -> None:
    _, store, teams = registry
    teams.refresh()
    gone = TeamRegistry(TeamsConfig(repo=str(tmp_path / "missing")), store)
    assert [t.team_id for t in gone.refresh()] == ["alpha"]
