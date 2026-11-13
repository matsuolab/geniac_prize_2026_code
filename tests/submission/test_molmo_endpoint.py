"""Where the fallback object localizer sends its requests.

When SAM3's language path finds nothing, ``molmo.py`` is what stands between the
agent and "object not found". Its endpoint used to be a module constant pointing at
a separate Molmo server — which does not exist inside an offline
submission container, so the fallback was absent rather than merely worse. One
137-task run made 2933 refused requests across 115 tasks before returning
``(None, None, None)`` each time.

The endpoint is configurable now so a container can point it at a vision-language
model it is already serving. These tests pin that it really is configurable, and
what an unset environment points at.
"""

from __future__ import annotations

import importlib
import sys

import pytest

MODULE = "capx.integrations.vision.molmo"


def _reload(monkeypatch: pytest.MonkeyPatch, **env: str):
    for key in ("CAPX_MOLMO_URL", "CAPX_MOLMO_MODEL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop(MODULE, None)
    return importlib.import_module(MODULE)


def test_the_endpoint_defaults_to_the_molmo_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    molmo = _reload(monkeypatch)
    assert molmo.SERVICE_URL == "http+unix://molmo.sock/v1"
    assert molmo.DEFAULT_MODEL == "allenai/Molmo2-8B"


def test_a_container_can_point_it_at_its_own_model_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molmo = _reload(
        monkeypatch,
        CAPX_MOLMO_URL="http+unix://model.sock/v1",
        CAPX_MOLMO_MODEL="Qwen/Qwen3.6-27B",
    )
    assert molmo.SERVICE_URL == "http+unix://model.sock/v1"
    assert molmo.DEFAULT_MODEL == "Qwen/Qwen3.6-27B"


def test_init_molmo_takes_the_configured_endpoint_without_being_told(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every call site does init_molmo() with no arguments, so the defaults are the
    only thing that can carry a redirect."""
    molmo = _reload(
        monkeypatch,
        CAPX_MOLMO_URL="http+unix://model.sock/v1",
        CAPX_MOLMO_MODEL="Qwen/Qwen3.6-27B",
    )
    import inspect

    defaults = inspect.signature(molmo.init_molmo).parameters
    assert defaults["base_url"].default == "http+unix://model.sock/v1"
    assert defaults["model_name"].default == "Qwen/Qwen3.6-27B"


def test_a_request_goes_to_the_configured_chat_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molmo = _reload(monkeypatch, CAPX_MOLMO_URL="http+unix://model.sock/v1")
    seen: dict[str, object] = {}

    class FakeSession:
        def post(self, url, json, headers, timeout):  # noqa: ANN001
            seen["url"] = url
            seen["model"] = json["model"]

            class Response:
                def raise_for_status(self) -> None:
                    pass

                @staticmethod
                def json() -> dict:
                    return {"choices": [{"message": {"content": "<point x=\"50\" y=\"50\">"}}]}

            return Response()

    monkeypatch.setattr(molmo.unix_http_client, "new_session", lambda: FakeSession())
    from PIL import Image

    detect = molmo.init_molmo()
    detect(Image.new("RGB", (8, 8)), objects=["hammer"])

    assert seen["url"] == "http+unix://model.sock/v1/chat/completions"


def test_a_tcp_endpoint_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    molmo = _reload(monkeypatch, CAPX_MOLMO_URL="http://127.0.0.1:8122/v1")
    with pytest.raises(ValueError, match="Unix sockets"):
        molmo.init_molmo()


# --- reading the reply -------------------------------------------------------
#
# Reaching the server was only half of it. Pointed at the container's own Qwen,
# 815 of 820 replies still parsed to no point at all: the parser read every
# untagged format on Molmo1's 0-100 scale and discarded 641 as out of range. The
# cases below are reply shapes taken verbatim from that run's log.


@pytest.fixture()
def parse(monkeypatch: pytest.MonkeyPatch):
    return _reload(monkeypatch)._parse_points


def test_molmo1_stays_on_its_own_scale(parse) -> None:
    """Small numbers are the one case that is genuinely 0-100."""
    assert parse('<point x="32.7" y="65.9">red square</point>') == ([(32.7, 65.9)], 100.0)


def test_molmo2_keeps_the_scale_it_declares(parse) -> None:
    assert parse('<points coords="1 0 332 683">cube</points>') == ([(332.0, 683.0)], 1000.0)


def test_a_coordinate_above_100_cannot_be_on_the_0_100_scale(parse) -> None:
    reply = '<points x1="641" y1="475" alt="stove_knob">stove_knob</points>'
    assert parse(reply) == ([(641.0, 475.0)], 1000.0)


def test_a_box_is_read_as_its_centre(parse) -> None:
    """Asked to point, the model answers with a bounding box about a third of the
    time. Its top-left corner is not where the object is."""
    assert parse("[413, 494, 435, 631]") == ([(424.0, 562.5)], 1000.0)
    assert parse('```json\n[{"bbox_2d": [413, 494, 435, 631], "label": "plate"}]\n```') == (
        [(424.0, 562.5)],
        1000.0,
    )


def test_a_json_point_is_a_point(parse) -> None:
    reply = '```json\n[\n\t{"point_2d": [339, 624], "label": "stove"}\n]\n```'
    assert parse(reply) == ([(339.0, 624.0)], 1000.0)


def test_a_refusal_yields_nothing(parse) -> None:
    """"There is none" has to stay an absent point, not a coordinate at the origin."""
    assert parse("There is none.")[0] == []
    assert parse("There are two mugs that could be described as white in the image.")[0] == []
