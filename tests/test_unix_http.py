"""The local model and tool servers listen only on Unix sockets, and their clients
reach them over ``http+unix://`` URLs (capx/utils/unix_http.py)."""

from __future__ import annotations

import re
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from capx.serving.api_servers import ApiServerSpec
from capx.utils import unix_http


@pytest.fixture
def socket_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fresh $CAPX_SOCKET_DIR short enough for socket paths, which tmp_path may not be."""
    base = Path(tempfile.mkdtemp(prefix="capx"))
    directory = base / "sockets"
    monkeypatch.setenv(unix_http.SOCKET_DIR_ENV, str(directory))
    yield directory
    shutil.rmtree(base, ignore_errors=True)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --- URLs --------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http+unix://model.sock/v1/chat/completions",
        "http+unix://%2Fhome%2Fme%2F.capx%2Fsockets%2Fsam3.sock/segment",
        "https://api.openai.com/v1/chat/completions",
    ],
)
def test_a_unix_socket_or_remote_https_url_is_allowed(url: str) -> None:
    assert unix_http.check_server_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8110/v1/chat/completions",
        "http://localhost:8114",
        "http://0.0.0.0:8000/v1",
        "http://example.com/v1",
        "http+unix:///v1/chat/completions",
        "model.sock/v1",
    ],
)
def test_any_other_url_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="Unix sockets"):
        unix_http.check_server_url(url)


def test_a_socket_url_round_trips_through_percent_encoding() -> None:
    url = unix_http.unix_url("/home/me/.capx/sockets/sam3.sock", "/segment")
    assert url == "http+unix://%2Fhome%2Fme%2F.capx%2Fsockets%2Fsam3.sock/segment"


def test_a_bare_socket_name_is_in_the_socket_dir(socket_dir: Path) -> None:
    assert unix_http.resolve_socket("sam3.sock") == socket_dir / "sam3.sock"
    assert unix_http.resolve_socket("/elsewhere/x.sock") == Path("/elsewhere/x.sock")


def test_the_socket_dir_defaults_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(unix_http.SOCKET_DIR_ENV, raising=False)
    assert unix_http.socket_dir() == Path.home() / ".capx" / "sockets"


def test_a_relative_socket_dir_is_taken_from_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(unix_http.SOCKET_DIR_ENV, "sockets")
    assert unix_http.socket_dir() == tmp_path / "sockets"
    assert unix_http.resolve_socket("model.sock") == tmp_path / "sockets" / "model.sock"


def test_a_socket_path_over_the_kernel_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="kernel limit"):
        unix_http.resolve_socket("/" + "d" * 120 + "/model.sock")


# --- the socket directory and file -------------------------------------------


def test_the_socket_dir_is_created_owner_only(socket_dir: Path) -> None:
    unix_http.ensure_private_dir(socket_dir)
    assert _mode(socket_dir) == 0o700


def test_a_socket_dir_others_can_enter_is_refused(socket_dir: Path) -> None:
    socket_dir.mkdir()
    socket_dir.chmod(0o755)
    with pytest.raises(PermissionError, match="chmod 700"):
        unix_http.bind_unix_socket("x.sock")


def test_a_bound_socket_is_owner_only(socket_dir: Path) -> None:
    with unix_http.bind_unix_socket("x.sock") as sock:
        path = Path(sock.getsockname())
        assert path == socket_dir / "x.sock"
        assert stat.S_ISSOCK(path.stat().st_mode)
        assert _mode(path) == 0o600
        assert _mode(socket_dir) == 0o700


def test_a_socket_left_by_a_dead_server_is_replaced(socket_dir: Path) -> None:
    unix_http.bind_unix_socket("x.sock").close()
    assert (socket_dir / "x.sock").exists()
    with unix_http.bind_unix_socket("x.sock"):
        assert unix_http.is_listening(socket_dir / "x.sock")


def test_a_live_socket_is_not_taken_over(socket_dir: Path) -> None:
    with unix_http.bind_unix_socket("x.sock"):
        with pytest.raises(OSError, match="already listening"):
            unix_http.bind_unix_socket("x.sock")


def test_a_file_that_is_not_a_socket_is_left_alone(socket_dir: Path) -> None:
    unix_http.ensure_private_dir(socket_dir)
    (socket_dir / "x.sock").write_text("keep me")
    with pytest.raises(FileExistsError):
        unix_http.bind_unix_socket("x.sock")
    assert (socket_dir / "x.sock").read_text() == "keep me"


# --- api_servers entries -----------------------------------------------------


def test_an_api_server_entry_names_its_socket() -> None:
    spec = ApiServerSpec.model_validate(
        {"_target_": "capx.serving.launch_sam3_server.main", "socket": "sam3.sock", "device": "cuda"}
    )
    assert spec.socket == "sam3.sock"


@pytest.mark.parametrize(
    "entry",
    [
        {"_target_": "capx.serving.launch_sam3_server.main", "port": 8114, "host": "127.0.0.1"},
        {"_target_": "capx.serving.launch_sam3_server.main", "socket": "s.sock", "port": 8114},
        {"_target_": "capx.serving.launch_sam3_server.main"},
    ],
)
def test_an_api_server_entry_without_a_socket_or_with_tcp_is_refused(entry: dict) -> None:
    with pytest.raises(ValidationError):
        ApiServerSpec.model_validate(entry)


def test_launch_servers_starts_each_server_on_its_socket() -> None:
    launch_servers = pytest.importorskip("capx.serving.launch_servers")
    servers = launch_servers.parse_servers_from_yaml("env_configs/gateway/franka_libero.yaml")
    assert {s["server"]: s["socket"] for s in servers} == {
        "sam3": "sam3.sock",
        "graspnet": "graspnet.sock",
        "pyroki": "pyroki.sock",
    }
    cmd = launch_servers._build_cmd(servers[0], workers=1)
    assert cmd[3:5] == ["--socket", "sam3.sock"]
    assert "--port" not in cmd and "--host" not in cmd


_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = sorted((_REPO / "env_configs").rglob("*.yaml"))
_LOCAL_TCP_URL = re.compile(r"https?://(127\.0\.0\.1|localhost|0\.0\.0\.0)\b")


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: str(p.relative_to(_REPO / "env_configs")))
def test_every_config_reaches_its_servers_over_unix_sockets(path: Path) -> None:
    text = path.read_text()
    assert not _LOCAL_TCP_URL.search(text)
    cfg = yaml.safe_load(text) or {}
    for entry in cfg.get("api_servers") or []:
        ApiServerSpec.model_validate(entry)
    for key in ("server_url", "visual_differencing_model_server_url"):
        if isinstance(cfg.get(key), str):
            unix_http.check_server_url(cfg[key])


def test_launch_servers_refuses_a_tcp_entry(tmp_path: Path) -> None:
    launch_servers = pytest.importorskip("capx.serving.launch_servers")
    config = tmp_path / "c.yaml"
    config.write_text(
        "api_servers:\n  - _target_: capx.serving.launch_sam3_server.main\n    port: 8114\n"
    )
    with pytest.raises(ValidationError, match="Unix sockets"):
        launch_servers.parse_servers_from_yaml(str(config))


# --- a real server -----------------------------------------------------------


def _echo_app():  # noqa: ANN202
    fastapi = pytest.importorskip("fastapi")
    app = fastapi.FastAPI()

    @app.post("/echo")
    def echo(body: dict) -> dict:
        return {"got": body}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat/completions")
    def chat(body: dict) -> dict:
        message = {"role": "assistant", "content": f"hello from {body['model']}"}
        return {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }

    return app


def serve_echo(socket: str) -> None:
    """An ``api_servers`` target, the shape of capx/serving's launchers."""
    unix_http.serve_app(_echo_app(), socket)


def _wait_for(path: Path, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not unix_http.is_listening(path):
        assert time.monotonic() < deadline, f"nothing listening on {path}"
        time.sleep(0.05)


@pytest.fixture
def echo_server(socket_dir: Path) -> Iterator[Path]:
    pytest.importorskip("uvicorn")
    app = _echo_app()
    thread = threading.Thread(target=unix_http.serve_app, args=(app, "echo.sock"), daemon=True)
    thread.start()
    path = socket_dir / "echo.sock"
    _wait_for(path)
    yield path


def test_a_client_reaches_a_uvicorn_server_over_its_socket(echo_server: Path) -> None:
    client = pytest.importorskip("capx.utils.unix_http_client")
    assert _mode(echo_server) == 0o600
    by_name = client.post("http+unix://echo.sock/echo", json={"a": 1}, timeout=5)
    assert by_name.status_code == 200 and by_name.json() == {"got": {"a": 1}}
    by_path = client.post(unix_http.unix_url(echo_server, "/echo"), json={"b": 2}, timeout=5)
    assert by_path.json() == {"got": {"b": 2}}
    assert client.get("http+unix://echo.sock/health", timeout=5).json() == {"status": "ok"}


def test_a_streamed_reply_reads_from_the_unix_socket(echo_server: Path) -> None:
    """capx.llm.client sets read deadlines on ``response.raw.connection.sock``."""
    client = pytest.importorskip("capx.utils.unix_http_client")
    reply = client.post("http+unix://echo.sock/echo", json={"x": 1}, timeout=5, stream=True)
    assert reply.raw.connection.sock.family == socket.AF_UNIX
    assert reply.json() == {"got": {"x": 1}}


def test_no_listener_is_a_connection_error(socket_dir: Path) -> None:
    requests = pytest.importorskip("requests")
    client = pytest.importorskip("capx.utils.unix_http_client")
    with pytest.raises(requests.ConnectionError):
        client.post("http+unix://nobody.sock/echo", json={}, timeout=1)


def test_the_client_refuses_tcp_before_connecting() -> None:
    client = pytest.importorskip("capx.utils.unix_http_client")
    with pytest.raises(ValueError, match="Unix sockets"):
        client.post("http://127.0.0.1:8114/segment", json={}, timeout=1)


def test_the_openai_sdk_reaches_a_proxy_over_its_socket(echo_server: Path) -> None:
    openai_client = pytest.importorskip("capx.utils.openai_client")
    client = openai_client.openai_client("http+unix://echo.sock/", "unused")
    reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert reply.choices[0].message.content == "hello from m"


def test_the_openai_sdk_keeps_https_and_refuses_tcp() -> None:
    openai_client = pytest.importorskip("capx.utils.openai_client")
    remote = openai_client.openai_client("https://openrouter.ai/api/v1/", "key")
    assert str(remote.base_url) == "https://openrouter.ai/api/v1/"
    with pytest.raises(ValueError, match="Unix sockets"):
        openai_client.openai_client("http://127.0.0.1:8110/", "key")


def test_the_sidecar_helper_posts_over_the_socket(echo_server: Path) -> None:
    serve_utils = pytest.importorskip("capx.utils.serve_utils")
    assert serve_utils.post_with_retries("http+unix://echo.sock/echo", {"k": "v"}) == {
        "got": {"k": "v"}
    }


def _runner() -> types.ModuleType:
    pytest.importorskip("uvicorn")
    if "capx.integrations" not in sys.modules:
        try:
            import capx.integrations  # noqa: F401
        except ImportError:
            import capx

            pkg = types.ModuleType("capx.integrations")
            pkg.__path__ = [str(Path(capx.__file__).parent / "integrations")]
            sys.modules["capx.integrations"] = pkg
    return pytest.importorskip("capx.envs.runner")


def test_the_runner_starts_an_api_server_and_waits_for_its_socket(socket_dir: Path) -> None:
    runner = _runner()
    client = pytest.importorskip("capx.utils.unix_http_client")
    procs = runner._start_api_servers(
        [{"_target_": f"{__name__}.serve_echo", "socket": "sidecar.sock"}], wait_timeout=60.0
    )
    try:
        assert len(procs) == 1
        assert unix_http.is_listening(socket_dir / "sidecar.sock")
        assert client.post("http+unix://sidecar.sock/echo", json={"r": 1}, timeout=5).json() == {
            "got": {"r": 1}
        }
        # A second config naming the same socket reuses the running server.
        assert runner._start_api_servers([{"_target_": "x.main", "socket": "sidecar.sock"}]) == []
    finally:
        runner._stop_api_servers(procs)
    assert not unix_http.is_listening(socket_dir / "sidecar.sock")
