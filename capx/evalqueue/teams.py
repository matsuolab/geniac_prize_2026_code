"""The registered teams.

The organizers create each team's ``<org>/capx-submission`` and register it by adding
``teams/prelim/<team_id>.yaml`` to this repository (PRELIM_RULEBOOK.md §6.1). The
``team_id`` is the team's HuggingFace org::

    team_id: team-alpha
    repo: team-alpha/capx-submission

The watcher reads the registry out of git on every poll, so a merged registration is
picked up without a restart. It keeps the last list it read in the store, where the
processes that do not fetch (the leaderboard, the weights check) read it.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import PurePosixPath

import yaml
from pydantic import ValidationError

from capx.evalqueue.config import HubTeam, TeamsConfig
from capx.evalqueue.models import describe_errors, hf_owner
from capx.evalqueue.store import EvalStore, write_json_atomic

_logger = logging.getLogger(__name__)

_SNAPSHOT = "teams.json"


class RegistryError(RuntimeError):
    """git could not fetch or read the registry."""


def parse_registry(files: dict[str, str]) -> tuple[list[HubTeam], list[str]]:
    """The teams in ``files`` (file name -> contents), and what is wrong with the rest.

    A file that does not parse, is not named after its ``team_id``, or whose ``team_id``
    is not its repo's org, is left out. So are all the teams whose orgs differ only in
    case: the Hub treats them as one org, and an org cannot belong to two teams.
    """
    teams: list[HubTeam] = []
    problems: list[str] = []
    for name, text in sorted(files.items()):
        try:
            team = HubTeam.model_validate(yaml.safe_load(text))
        except yaml.YAMLError as exc:
            problems.append(f"{name}: not YAML ({str(exc).splitlines()[0]})")
            continue
        except ValidationError as exc:
            problems.append(f"{name}: {describe_errors(exc)}")
            continue
        if name != f"{team.team_id}.yaml":
            problems.append(f"{name}: must be named {team.team_id}.yaml")
            continue
        org = team.repo.split("/", 1)[0]
        if team.team_id != org:
            problems.append(f"{name}: team_id must be the repo's org, {org}")
            continue
        teams.append(team)
    by_org: dict[str, list[str]] = {}
    for team in teams:
        by_org.setdefault(hf_owner(team.repo), []).append(team.team_id)
    shared = {org for org, team_ids in by_org.items() if len(team_ids) > 1}
    for org in sorted(shared):
        problems.append(f"org {org!r} is named by more than one team: {', '.join(by_org[org])}")
    return [team for team in teams if hf_owner(team.repo) not in shared], problems


def team_orgs(teams: list[HubTeam]) -> dict[str, str]:
    """HuggingFace org -> team_id."""
    return {team.repo.split("/", 1)[0]: team.team_id for team in teams}


def registered_teams(store: EvalStore) -> list[HubTeam]:
    """The teams the watcher last read from the registry; none before its first read."""
    try:
        rows = json.loads((store.state / _SNAPSHOT).read_text())
    except (OSError, ValueError):
        return []
    return [HubTeam.model_validate(row) for row in rows]


def save_registered_teams(store: EvalStore, teams: list[HubTeam]) -> None:
    write_json_atomic(store.state / _SNAPSHOT, [team.model_dump() for team in teams])


class TeamRegistry:
    """Reads the registry with one shallow ``git fetch`` per refresh, into a bare
    repository under the store."""

    def __init__(self, config: TeamsConfig, store: EvalStore) -> None:
        self._config = config
        self._store = store
        self._git_dir = store.state / "teams.git"
        self._problems: list[str] = []

    def refresh(self) -> list[HubTeam]:
        """The teams registered on ``ref`` now, or the last ones read if git fails:
        a GitHub outage must not unregister everyone."""
        try:
            files = self._fetch()
        except RegistryError as exc:
            _logger.warning("cannot read the team registry (%s); using the last one read", exc)
            return registered_teams(self._store)
        teams, problems = parse_registry(files)
        if problems != self._problems:
            for problem in problems:
                _logger.warning("team registry: %s", problem)
            self._problems = problems
        save_registered_teams(self._store, teams)
        return teams

    def _fetch(self) -> dict[str, str]:
        if not (self._git_dir / "HEAD").exists():
            _git("init", "--bare", "--quiet", str(self._git_dir))
        git_dir = ("--git-dir", str(self._git_dir))
        _git(*git_dir, "fetch", "--quiet", "--depth", "1", "--no-tags",
             self._config.repo, self._config.ref)  # fmt: skip
        directory = self._config.path.strip("/")
        listing = _git(*git_dir, "ls-tree", "--name-only", "FETCH_HEAD", f"{directory}/")
        return {
            PurePosixPath(path).name: _git(*git_dir, "show", f"FETCH_HEAD:{path}")
            for path in listing.splitlines()
            if path.endswith(".yaml")
        }


def _git(*args: str) -> str:
    try:
        done = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            # Never wait on a password prompt nobody will answer.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RegistryError(f"git {args[-1]}: {exc}") from exc
    if done.returncode != 0:
        detail = done.stderr.strip().splitlines()
        raise RegistryError(f"git exited {done.returncode}: {detail[-1] if detail else ''}")
    return done.stdout


__all__ = [
    "RegistryError",
    "TeamRegistry",
    "parse_registry",
    "registered_teams",
    "save_registered_teams",
    "team_orgs",
]
