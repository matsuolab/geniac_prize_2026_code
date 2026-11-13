"""Each reference task config starts the servers its control APIs call, and no others.

The two-arm lift config once started SAM3, GraspNet and PyRoKi while
``FrankaTwoArmLiftApi`` finds the handles with OWL-ViT then SAM2: every perception
call in every two-arm trial hit a closed port, retried for ~23 s and failed.

The table is read off each API's constructor (the ``init_*`` clients it builds; the
Molmo client goes to the model server itself, see singularity/agent0-entrypoint.sh).
A config naming an API missing from it fails here, so it is kept up to date.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from capx.serving.api_servers import ApiServerSpec
from capx.submission.interfaces import (
    GATEWAY_CONFIG_DIR,
    load_submission_config,
    load_task_run_config,
)

_SAM3 = "capx.serving.launch_sam3_server.main"
_SAM2 = "capx.serving.launch_sam2_server.main"
_OWLVIT = "capx.serving.launch_owlvit_server.main"
_GRASPNET = "capx.serving.launch_contact_graspnet_server.main"
_PYROKI = "capx.serving.launch_pyroki_server.main"

_SERVERS_BY_API = {
    # FrankaControlApiReduced(use_sam3=True): SAM3, GraspNet, pyroki IK and trajopt.
    "FrankaControlApiReducedSkillLibrary": {_SAM3, _GRASPNET, _PYROKI},
    # FrankaLiberoApi(use_sam3=True): SAM3, GraspNet, pyroki IK.
    "FrankaLiberoApi": {_SAM3, _GRASPNET, _PYROKI},
    "FrankaHandoverApi": {_SAM3, _GRASPNET, _PYROKI},
    # Solves IK in process with JAX.
    "FrankaTwoArmLiftApi": {_OWLVIT, _SAM2, _GRASPNET},
    # R1ProControlApi(use_sam3=True): its IK is in process too.
    "R1ProControlApi": {_SAM3, _GRASPNET},
}

_TASK_CONFIGS = sorted(p for p in GATEWAY_CONFIG_DIR.glob("*.yaml") if p.name != "runner.yaml")


@pytest.mark.parametrize("path", _TASK_CONFIGS, ids=lambda p: p.name)
def test_a_task_config_starts_exactly_the_servers_its_apis_call(path: Path) -> None:
    task_cfg = load_task_run_config(str(path))
    needed = set().union(*(_SERVERS_BY_API[api] for api in task_cfg.apis))
    started = {server["_target_"] for server in task_cfg.api_servers}
    assert started == needed


# The sockets the clients (capx/integrations/vision, capx/integrations/motion) default to.
_SOCKET_BY_SERVER = {
    _SAM3: "sam3.sock",
    _SAM2: "sam2.sock",
    _OWLVIT: "owlvit.sock",
    _GRASPNET: "graspnet.sock",
    _PYROKI: "pyroki.sock",
}


@pytest.mark.parametrize("path", _TASK_CONFIGS, ids=lambda p: p.name)
def test_each_server_listens_on_the_socket_its_client_calls(path: Path) -> None:
    for server in load_task_run_config(str(path)).api_servers:
        spec = ApiServerSpec.model_validate(server)
        assert spec.socket == _SOCKET_BY_SERVER[spec.target]


def test_the_runner_config_reaches_the_model_over_a_unix_socket() -> None:
    cfg = load_submission_config(str(GATEWAY_CONFIG_DIR / "runner.yaml"))
    assert cfg.server_url.startswith("http+unix://")
