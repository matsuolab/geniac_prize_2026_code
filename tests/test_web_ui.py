"""The interactive web UI runs generated code as the user who started it.

So it listens on 127.0.0.1 unless told otherwise, refuses API and WebSocket requests
without its access token (and never puts the token in a cookie, which every port on the
host would receive), does not hand out files outside its build, and takes configs only
from env_configs/.

Needs fastapi and the config loader's dependencies, so it skips on a bare checkout.
"""

from __future__ import annotations

import asyncio
import http.server
import threading
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("omegaconf")
pytest.importorskip("cv2")
pytest.importorskip("requests")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from capx.web import server, session_manager  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TOKEN_PROTOCOL = f"{server.TOKEN_SUBPROTOCOL_PREFIX}{TOKEN}"


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session_manager, "_manager", None)
    (tmp_path / "env_configs").mkdir()
    (tmp_path / "env_configs" / "demo.yaml").write_text(
        "env:\n  _target_: demo.Env\n  cfg:\n    prompt: stack the cubes\n"
    )
    (tmp_path / "outside.yaml").write_text("env:\n  cfg: {}\n")
    (tmp_path / "secret.txt").write_text("secret")
    (tmp_path / "dist" / "assets").mkdir(parents=True)
    (tmp_path / "dist" / "index.html").write_text("index")
    (tmp_path / "dist" / "assets" / "app.js").write_text("app")
    return tmp_path


def _client(workdir: Path, **kwargs: Any) -> TestClient:
    return TestClient(server.create_app(access_token=TOKEN, frontend_dist=workdir / "dist", **kwargs))


def test_api_and_websockets_need_the_token(workdir: Path) -> None:
    client = _client(workdir)
    assert client.get("/api/configs").status_code == 401
    assert client.get("/api/configs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(f"/api/configs?token={TOKEN}").status_code == 401
    response = client.get("/api/configs", headers=AUTH)
    assert response.json() == {"configs": ["env_configs/demo.yaml"]}
    assert "set-cookie" not in response.headers
    # CORS preflights for the dev server carry no token and are answered before the check.
    preflight = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    assert client.options("/api/start-trial", headers=preflight).status_code == 200

    with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect("/ws/any"):
        pass
    assert refused.value.code == 1008
    # With the token the request reaches the endpoint, which knows no such session.
    with pytest.raises(WebSocketDisconnect) as unknown, client.websocket_connect(
        "/ws/any", subprotocols=[TOKEN_PROTOCOL]
    ):
        pass
    assert unknown.value.code == 4004


@pytest.mark.parametrize("host", ["localhost/?", "localhost/#", "x/?a="])
def test_a_crafted_host_header_does_not_skip_the_token(workdir: Path, host: str) -> None:
    client = _client(workdir)
    assert client.get("/api/configs", headers={"host": host}).status_code == 401
    assert client.get("/viser-proxy/wrong/", headers={"host": host}).status_code == 401
    with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect(
        "/ws/any", headers={"host": host}
    ):
        pass
    assert refused.value.code == 1008


def test_websocket_accepts_the_token_subprotocol(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_forever() -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "run_trial_async", lambda session, args: run_forever())
    with _client(workdir) as client:
        body = {"config_path": "env_configs/demo.yaml"}
        session_id = client.post("/api/start-trial", json=body, headers=AUTH).json()["session_id"]
        with client.websocket_connect(f"/ws/{session_id}", subprotocols=[TOKEN_PROTOCOL]) as ws:
            assert ws.accepted_subprotocol == TOKEN_PROTOCOL
            assert ws.receive_json()["type"] == "state_update"


def test_viser_proxy_takes_its_own_key_and_nothing_else(workdir: Path) -> None:
    client = _client(workdir)
    viser_url = client.get("/api/viser-url", headers=AUTH).json()["url"]
    key = viser_url.split("/")[2]
    assert client.get("/viser-proxy/wrong/").status_code == 401
    assert client.get("/viser-proxy/", headers=AUTH).status_code == 401
    assert client.get(viser_url).status_code == 503  # authorized; no Viser server is running
    assert client.get("/api/configs", headers={"Authorization": f"Bearer {key}"}).status_code == 401


@pytest.fixture
def html_server() -> Iterator[int]:
    """A local HTTP server that answers every GET with a page, like a Viser server."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            body = b"<script>localStorage.getItem('capx-webui-token')</script>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()


def test_proxied_viser_pages_are_sandboxed(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, html_server: int
) -> None:
    monkeypatch.setattr(session_manager.SessionManager, "viser_port", lambda self: html_server)
    client = _client(workdir)
    viser_url = client.get("/api/viser-url", headers=AUTH).json()["url"]
    for path in (viser_url, viser_url + "assets/index.js"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-security-policy"] == "sandbox allow-scripts"


def test_viser_port_comes_from_the_trial_env_not_a_probe() -> None:
    manager = session_manager.SessionManager()
    assert manager.viser_port() is None
    session = asyncio.run(manager.create_session())
    session.env = types.SimpleNamespace(
        low_level_env=types.SimpleNamespace(viser_server=types.SimpleNamespace(get_port=lambda: 8123))
    )
    assert manager.viser_port() == 8123


@pytest.mark.parametrize("path", ["/..%2fsecret.txt", "/%2e%2e/secret.txt", "/..%2f..%2fsecret.txt"])
def test_frontend_route_serves_nothing_outside_the_build(workdir: Path, path: str) -> None:
    client = _client(workdir)
    assert client.get(path).status_code == 401
    assert client.get(path, headers=AUTH).text == "index"


def test_frontend_route_ignores_absolute_paths(workdir: Path) -> None:
    client = _client(workdir)
    path = "/%2F" + str(workdir / "secret.txt").lstrip("/")
    assert client.get(path).status_code == 401
    assert client.get(path, headers=AUTH).text == "index"


def test_only_the_built_frontend_is_public(workdir: Path) -> None:
    client = _client(workdir)
    assert client.get("/").text == "index"
    assert client.get("/index.html").text == "index"
    assert client.get("/assets/app.js").text == "app"
    assert client.post("/").status_code == 401
    assert client.get("/some/client/route").status_code == 401
    assert client.get("/some/client/route", headers=AUTH).text == "index"


def test_the_token_is_required_when_the_app_is_mounted_under_a_prefix(workdir: Path) -> None:
    parent = FastAPI()
    parent.mount("/capx", server.create_app(access_token=TOKEN, frontend_dist=workdir / "dist"))
    client = TestClient(parent)
    assert client.get("/capx/api/configs").status_code == 401
    assert client.get("/capx/api/configs", headers=AUTH).status_code == 200
    assert client.get("/capx/viser-proxy/wrong/").status_code == 401
    assert client.get("/capx/").text == "index"
    with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect("/capx/ws/any"):
        pass
    assert refused.value.code == 1008


@pytest.mark.parametrize("config_path", ["outside.yaml", "env_configs/../outside.yaml", "/etc/hostname"])
def test_configs_outside_env_configs_are_refused(workdir: Path, config_path: str) -> None:
    client = _client(workdir)
    for route in ("/api/load-config", "/api/start-trial"):
        response = client.post(route, json={"config_path": config_path}, headers=AUTH)
        assert response.status_code == 403, route


def test_the_launch_config_is_allowed_wherever_it_is(workdir: Path) -> None:
    client = _client(workdir, default_config_path="outside.yaml")
    response = client.post("/api/load-config", json={"config_path": "outside.yaml"}, headers=AUTH)
    assert response.status_code == 200
    response = client.post("/api/load-config", json={"config_path": "env_configs/demo.yaml"}, headers=AUTH)
    assert response.json()["task_prompt"] == "stack the cubes"


@pytest.mark.parametrize(
    ("env_host", "host", "bound", "url_host"),
    [
        (None, None, "127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", None, "0.0.0.0", "127.0.0.1"),
        ("0.0.0.0", "::1", "::1", "[::1]"),
    ],
)
def test_serve_binds_loopback_unless_told_and_prints_the_literal_address(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env_host: str | None,
    host: str | None,
    bound: str,
    url_host: str,
) -> None:
    monkeypatch.delenv("CAPX_WEB_UI_PUBLIC_URL", raising=False)
    if env_host is None:
        monkeypatch.delenv("CAPX_WEB_UI_HOST", raising=False)
    else:
        monkeypatch.setenv("CAPX_WEB_UI_HOST", env_host)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    server.serve(server.create_app(access_token=TOKEN, frontend_dist=workdir / "dist"), port=8200, host=host)
    assert calls == [{"host": bound, "port": 8200}]
    # Never "localhost": it may resolve to ::1, where another user could be listening.
    assert f"http://{url_host}:8200/#token={TOKEN}" in capsys.readouterr().out


def test_serve_prints_the_published_address_of_a_container(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # docker/run.sh publishes container port 8200 on a random host port; a URL naming
    # host port 8200 could reach another user's server, which would read the token.
    monkeypatch.setenv("CAPX_WEB_UI_HOST", "0.0.0.0")
    monkeypatch.setenv("CAPX_WEB_UI_PUBLIC_URL", "http://127.0.0.1:43210/")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    server.serve(server.create_app(access_token=TOKEN, frontend_dist=workdir / "dist"), port=8200)
    assert calls == [{"host": "0.0.0.0", "port": 8200}]
    out = capsys.readouterr().out
    assert f"http://127.0.0.1:43210/#token={TOKEN}" in out
    assert ":8200/" not in out
