"""FastAPI server for CaP-X interactive web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request as UrlRequest

import tyro
import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette._utils import get_route_path  # the path routing matches, root_path removed
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from capx.envs.configs.instantiate import instantiate
from capx.utils.launch_utils import _load_config
from capx.web.async_trial_runner import LaunchArgsCompat, run_trial_async
from capx.web.models import (
    ConfigListResponse,
    InjectPromptCommand,
    LoadConfigRequest,
    LoadConfigResponse,
    SessionState,
    SessionStatusResponse,
    StartTrialRequest,
    StartTrialResponse,
    StateUpdateEvent,
    StopCommand,
    StopTrialRequest,
    StopTrialResponse,
)
from capx.web.session_manager import Session, get_session_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "web-ui" / "dist"
CONFIGS_ROOT = Path("env_configs")
TOKEN_SUBPROTOCOL_PREFIX = "capx-token."
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _token_subprotocol(websocket: WebSocket) -> str | None:
    return next(
        (p for p in websocket.scope.get("subprotocols", []) if p.startswith(TOKEN_SUBPROTOCOL_PREFIX)),
        None,
    )


class _TokenAuthMiddleware:
    """Require the access token on every request, except for the built frontend's files
    (GET/HEAD) and the Viser proxy under its own key.

    The token comes in an ``Authorization: Bearer`` header, or, for a browser WebSocket,
    as the subprotocol ``capx-token.<token>``. The browser keeps it in this origin's
    storage: a cookie would also be sent to every other port on the host. The Viser
    iframe and client can send neither, so the proxy lives under
    ``/viser-proxy/<viser key>/``, a separate key that grants only the 3D view.
    """

    def __init__(self, app: ASGIApp, token: str, viser_key: str, frontend_root: Path | None) -> None:
        self.app = app
        self.token = token.encode()
        self.viser_key = viser_key.encode()
        self.frontend_root = frontend_root

    @staticmethod
    def _matches(value: str | None, secret: bytes) -> bool:
        return value is not None and secrets.compare_digest(value.encode(), secret)

    def _is_frontend_file(self, route_path: str) -> bool:
        if self.frontend_root is None:
            return False
        if route_path in ("", "/"):
            return True
        file_path = (self.frontend_root / route_path.lstrip("/")).resolve()
        return file_path.is_relative_to(self.frontend_root) and file_path.is_file()

    def _allowed(self, conn: HTTPConnection) -> bool:
        # Decided on the path routing uses: not conn.url (rebuilt from the Host header),
        # and without the root_path the app may be mounted under.
        route_path = get_route_path(conn.scope)
        segments = route_path.split("/")
        if len(segments) > 2 and segments[1] == "viser-proxy":
            return self._matches(segments[2], self.viser_key)
        if conn.scope.get("method") in ("GET", "HEAD") and self._is_frontend_file(route_path):
            return True
        scheme, _, credentials = conn.headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and self._matches(credentials, self.token):
            return True
        protocols = conn.scope.get("subprotocols", [])
        return any(
            self._matches(p.removeprefix(TOKEN_SUBPROTOCOL_PREFIX), self.token)
            for p in protocols
            if p.startswith(TOKEN_SUBPROTOCOL_PREFIX)
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or self._allowed(HTTPConnection(scope)):
            await self.app(scope, receive, send)
        elif scope["type"] == "websocket":
            await WebSocketClose(code=1008, reason="Missing or wrong access token")(scope, receive, send)
        else:
            detail = "Open the URL with #token=... that the CaP-X web UI printed when it started."
            await JSONResponse({"detail": detail}, status_code=401)(scope, receive, send)


# Proxied Viser pages run in an opaque origin, so they cannot read the token from this
# origin's storage. Its module scripts are then cross-origin loads, hence the CORS header.
_VISER_PROXY_HEADERS = {
    "Content-Security-Policy": "sandbox allow-scripts",
    "Access-Control-Allow-Origin": "*",
}


def create_app(
    *,
    default_config_path: str | None = None,
    access_token: str | None = None,
    frontend_dist: Path = FRONTEND_DIST,
) -> FastAPI:
    """Create and configure the FastAPI application.

    /api and /ws need ``access_token`` (a random one when not given; it is kept in
    ``app.state.access_token``). Configs are accepted only from under ``env_configs/``,
    plus ``default_config_path``.
    """
    app = FastAPI(
        title="CaP-X Interactive Web UI",
        description="Real-time interactive interface for CaP-X robot code execution",
        version="1.0.0",
    )
    app.state.access_token = access_token or secrets.token_urlsafe(32)
    viser_key = secrets.token_urlsafe(16)
    frontend_root = frontend_dist.resolve() if frontend_dist.exists() else None
    app.add_middleware(
        _TokenAuthMiddleware,
        token=app.state.access_token,
        viser_key=viser_key,
        frontend_root=frontend_root,
    )

    default_config = (
        Path(default_config_path).expanduser().resolve() if default_config_path else None
    )

    def allowed_config_path(config_path: str) -> str:
        resolved = Path(config_path).expanduser().resolve()
        if resolved != default_config and not resolved.is_relative_to(CONFIGS_ROOT.resolve()):
            raise HTTPException(status_code=403, detail=f"Configs must be under {CONFIGS_ROOT}/")
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
        return config_path

    # CORS for frontend dev server (added last, so it answers preflights before the token check)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========================================================================
    # REST API Endpoints
    # ========================================================================

    @app.get("/api/default-config")
    async def get_default_config():
        """Return the default config path passed via launch.py (if any).

        When a config_path was provided on the CLI, ``auto_start`` is set to
        ``True`` so the frontend can kick off the trial immediately after load.
        """
        return {
            "config_path": default_config_path,
            "auto_start": default_config_path is not None,
        }

    @app.get("/api/configs", response_model=ConfigListResponse)
    async def list_configs():
        """List available YAML config files from all environment directories."""
        if not CONFIGS_ROOT.exists():
            return ConfigListResponse(configs=[])

        configs = []
        for yaml_file in CONFIGS_ROOT.rglob("*.yaml"):
            # Skip hillclimb subdirectories (internal experiment configs)
            if "hillclimb" in yaml_file.parts:
                continue
            configs.append(str(yaml_file.relative_to(".")))

        configs.sort()
        return ConfigListResponse(configs=configs)

    @app.get("/api/viser-url")
    async def get_viser_url():
        """Where the frontend finds the Viser proxy (its path carries the Viser key)."""
        return {"url": f"/viser-proxy/{viser_key}/"}

    @app.post("/api/load-config", response_model=LoadConfigResponse)
    async def load_config(request: LoadConfigRequest):
        """Load and validate a YAML config file."""
        config_path = allowed_config_path(request.config_path)

        try:
            # Create a minimal args object for _load_config
            @dataclass
            class MinimalArgs:
                config_path: str
                server_url: str = "http+unix://model.sock/chat/completions"
                model: str = "google/gemini-3.1-pro-preview"
                temperature: float = 1.0
                max_tokens: int = 20480
                reasoning_effort: str = "medium"
                api_key: str | None = None
                use_visual_feedback: bool | None = None
                use_img_differencing: bool | None = None
                visual_differencing_model: str | None = "google/gemini-3.1-pro-preview"
                visual_differencing_model_server_url: str | None = "http+unix://model.sock/chat/completions"
                visual_differencing_model_api_key: str | None = None
                total_trials: int | None = None
                num_workers: int | None = None
                record_video: bool | None = None
                output_dir: str | None = None
                debug: bool = False
                use_oracle_code: bool | None = None
                use_parallel_ensemble: bool | None = None
                use_video_differencing: bool | None = None
                use_wrist_camera: bool | None = None
                use_multimodel: bool | None = None
                web_ui: bool | None = None
                web_ui_port: int | None = None

            args = MinimalArgs(config_path=config_path)
            env_factory, config, _ = await asyncio.to_thread(_load_config, args)

            # Extract task prompt (no length limit - UI handles display)
            task_prompt = env_factory.get("cfg", {}).get("prompt", "")

            return LoadConfigResponse(
                status="loaded",
                config_summary={
                    "record_video": config.get("record_video"),
                    "output_dir": config.get("output_dir"),
                    "use_visual_feedback": config.get("use_visual_feedback"),
                    "use_img_differencing": config.get("use_img_differencing"),
                    "total_trials": config.get("total_trials"),
                },
                task_prompt=task_prompt,
            )
        except Exception as e:
            logger.exception(f"Failed to load config: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/start-trial", response_model=StartTrialResponse)
    async def start_trial(request: StartTrialRequest):
        """Start a new trial execution."""
        config_path = allowed_config_path(request.config_path)

        manager = get_session_manager()

        # Create new session
        session = await manager.create_session()

        try:
            # Build args for config loading
            @dataclass
            class LoadArgs:
                config_path: str
                server_url: str
                model: str
                temperature: float
                max_tokens: int
                reasoning_effort: str = "medium"
                api_key: str | None = None
                use_visual_feedback: bool | None = None
                use_img_differencing: bool | None = None
                visual_differencing_model: str | None = None
                visual_differencing_model_server_url: str | None = None
                visual_differencing_model_api_key: str | None = None
                total_trials: int | None = 1
                num_workers: int | None = 1
                record_video: bool | None = None
                output_dir: str | None = None
                debug: bool = False
                use_oracle_code: bool | None = None
                use_parallel_ensemble: bool | None = None
                use_video_differencing: bool | None = None
                use_wrist_camera: bool | None = None
                use_multimodel: bool | None = None
                web_ui: bool | None = None
                web_ui_port: int | None = None

            load_args = LoadArgs(
                config_path=config_path,
                server_url=request.server_url,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_visual_feedback=request.use_visual_feedback,
                use_img_differencing=request.use_img_differencing,
                visual_differencing_model=request.visual_differencing_model,
                visual_differencing_model_server_url=request.visual_differencing_model_server_url,
            )
            env_factory, config, _ = await asyncio.to_thread(_load_config, load_args)

            # Process output_dir like launch.py does - add model name to path and create directory
            if config.get("output_dir"):
                from datetime import datetime
                # Add model name and timestamp to output path
                base_dir = config["output_dir"]
                model_slug = request.model.replace("/", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_out_dir = f"{base_dir}/{model_slug}/{timestamp}"
                Path(new_out_dir).mkdir(parents=True, exist_ok=True)
                config["output_dir"] = new_out_dir
                logger.info(f"Output directory set to: {new_out_dir}")

            # Store in session
            session.config_path = config_path
            session.config = config
            session.env_factory = env_factory
            session.state = SessionState.LOADING_CONFIG

            # Build launch args for trial runner
            trial_args = LaunchArgsCompat(
                model=request.model,
                server_url=request.server_url,
                api_key=None,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                reasoning_effort="medium",
                debug=False,
                visual_differencing_model=request.visual_differencing_model,
                visual_differencing_model_server_url=request.visual_differencing_model_server_url,
                visual_differencing_model_api_key=None,
            )

            # Set the initial settings on session (can be changed during trial)
            session.await_user_input_each_turn = request.await_user_input_each_turn
            session.execution_timeout = request.execution_timeout

            # Start trial in background task
            session.task = asyncio.create_task(
                run_trial_async(
                    session=session,
                    args=trial_args,
                )
            )

            return StartTrialResponse(
                session_id=session.session_id,
                status="started",
            )

        except Exception as e:
            logger.exception(f"Failed to start trial: {e}")
            await manager.remove_session(session.session_id)
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/stop", response_model=StopTrialResponse)
    async def stop_trial(request: StopTrialRequest):
        """Stop a running trial."""
        manager = get_session_manager()
        success = await manager.stop_session(request.session_id)

        if not success:
            raise HTTPException(status_code=404, detail="Session not found or not running")

        return StopTrialResponse(status="stopped")

    @app.post("/api/inject-prompt")
    async def inject_prompt(session_id: str, text: str):
        """Inject user prompt text into a running session."""
        manager = get_session_manager()
        success = await manager.inject_prompt(session_id, text)

        if not success:
            raise HTTPException(status_code=400, detail="Cannot inject: session not awaiting input")

        return {"status": "injected"}

    @app.get("/api/session/{session_id}", response_model=SessionStatusResponse)
    async def get_session_status(session_id: str):
        """Get the status of a session."""
        manager = get_session_manager()
        session = await manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionStatusResponse(
            session_id=session.session_id,
            state=session.state,
            current_block_index=session.current_block_index,
            total_code_blocks=session.total_code_blocks,
            num_regenerations=session.num_regenerations,
        )

    @app.get("/api/sessions")
    async def list_sessions():
        """List all active sessions."""
        manager = get_session_manager()
        return {"sessions": manager.list_sessions()}

    @app.get("/api/active-session")
    async def get_active_session():
        """Get the currently active session (if any).

        Useful for reconnecting after page refresh.
        """
        manager = get_session_manager()
        session = manager.get_active_session()
        if session:
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "config_path": session.config_path,
            }
        return {"session_id": None}

    # ========================================================================
    # WebSocket Endpoint
    # ========================================================================

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket connection for real-time trial updates."""
        manager = get_session_manager()
        session = await manager.get_session(session_id)

        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept(subprotocol=_token_subprotocol(websocket))
        session.websockets.append(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

        # Replay event history so reconnecting clients see all past messages
        if session.event_history:
            logger.info(f"Replaying {len(session.event_history)} events for session {session_id}")
            for event_json in session.event_history:
                try:
                    await websocket.send_text(event_json)
                except Exception:
                    break

        # Send current state
        await websocket.send_text(
            StateUpdateEvent(
                session_id=session_id,
                state=session.state,
            ).model_dump_json()
        )

        try:
            while True:
                # Receive commands from client
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    msg_type = message.get("type")

                    if msg_type == "stop":
                        logger.info(f"Stop command received for session {session_id}")
                        await manager.stop_session(session_id)

                    elif msg_type == "inject_prompt":
                        text = message.get("text", "")
                        if text:
                            await manager.inject_prompt(session_id, text)
                            logger.info(f"Injected prompt: {text[:50]}...")

                    elif msg_type == "resume":
                        # Put empty string to unblock the queue wait
                        if session.state == SessionState.AWAITING_USER_INPUT:
                            await session.user_injection_queue.put("")

                    elif msg_type == "update_settings":
                        # Update session settings dynamically during a trial
                        if "await_user_input_each_turn" in message:
                            session.await_user_input_each_turn = message["await_user_input_each_turn"]
                            logger.info(f"Updated await_user_input_each_turn to {session.await_user_input_each_turn}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {data}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        finally:
            if websocket in session.websockets:
                session.websockets.remove(websocket)
            # Check if session should be cleaned up
            await manager.on_websocket_disconnect(session_id)

    # ========================================================================
    # Viser reverse proxy  (avoids port-forwarding issues)
    # ========================================================================

    async def _proxy_viser_http(path: str = "", query: str = "") -> Response:
        """Forward an HTTP request to the Viser server of the current trial's env."""
        port = get_session_manager().viser_port()
        if port is None:
            return Response(
                content="Viser server not available — is the trial running?",
                status_code=503,
                headers=_VISER_PROXY_HEADERS,
            )
        url = f"http://localhost:{port}/{path}"
        if query:
            url += f"?{query}"
        try:
            req = UrlRequest(url)
            resp = await asyncio.to_thread(urlopen, req, None, 5)
            content = await asyncio.to_thread(resp.read)
            content_type = resp.headers.get(
                "Content-Type", "application/octet-stream"
            )
            return Response(content=content, media_type=content_type, headers=_VISER_PROXY_HEADERS)
        except Exception as exc:
            return Response(content=f"Proxy error: {exc}", status_code=502, headers=_VISER_PROXY_HEADERS)

    # The middleware has checked the key in these paths.
    @app.api_route("/viser-proxy/{key}", methods=["GET", "HEAD"])
    async def proxy_viser_root(request: Request, key: str):
        """Proxy the Viser root page (no trailing slash)."""
        return await _proxy_viser_http("", str(request.url.query))

    @app.api_route("/viser-proxy/{key}/{path:path}", methods=["GET", "HEAD"])
    async def proxy_viser_path(request: Request, key: str, path: str):
        """Proxy Viser sub-paths (assets, hdri, etc.)."""
        return await _proxy_viser_http(path, str(request.url.query))

    @app.websocket("/viser-proxy/{key}")
    async def proxy_viser_ws(websocket: WebSocket, key: str):
        """Reverse-proxy WebSocket connections to the local Viser server.

        The Viser client constructs the WS URL from ``window.location.href``
        (replacing http→ws and stripping the trailing slash), so the iframe at
        ``/viser-proxy/<key>/`` connects to ``ws://host/viser-proxy/<key>``.
        """
        port = get_session_manager().viser_port()
        if port is None:
            await websocket.close(code=1013, reason="Viser not running")
            return

        # Forward subprotocols (Viser expects "viser-v<VERSION>")
        proto_header = websocket.headers.get("sec-websocket-protocol", "")
        client_protocols = [
            s.strip() for s in proto_header.split(",") if s.strip()
        ]

        import websockets
        from websockets.asyncio.client import connect as ws_connect

        try:
            viser_ws = await ws_connect(
                f"ws://localhost:{port}/",
                subprotocols=(
                    [websockets.Subprotocol(p) for p in client_protocols]
                    if client_protocols
                    else None
                ),
                compression=None,
                max_size=2**24,  # 16 MiB — Viser can send large scene blobs
            )
        except Exception as exc:
            logger.warning(f"Could not connect to Viser WS: {exc}")
            await websocket.close(code=1013, reason=str(exc))
            return

        # Accept the browser connection with Viser's selected subprotocol
        await websocket.accept(subprotocol=viser_ws.subprotocol)

        async def _client_to_viser() -> None:
            try:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in msg and msg["bytes"]:
                        await viser_ws.send(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        await viser_ws.send(msg["text"])
            except (WebSocketDisconnect, Exception):
                pass

        async def _viser_to_client() -> None:
            try:
                async for data in viser_ws:
                    if isinstance(data, bytes):
                        await websocket.send_bytes(data)
                    else:
                        await websocket.send_text(data)
            except Exception:
                pass

        try:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_client_to_viser()),
                    asyncio.create_task(_viser_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
        except Exception as exc:
            logger.warning(f"Viser WS proxy error: {exc}")
        finally:
            await viser_ws.close()
            try:
                await websocket.close()
            except Exception:
                pass

    # ========================================================================
    # Static file serving for frontend (production)
    # ========================================================================

    if frontend_root is not None:
        app.mount("/assets", StaticFiles(directory=frontend_root / "assets"), name="assets")

        @app.get("/{path:path}")
        async def serve_frontend(path: str):
            # Serve built files, and index.html for anything else (SPA routing).
            file_path = (frontend_root / path).resolve()
            if file_path.is_relative_to(frontend_root) and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_root / "index.html")

    return app


def serve(app: FastAPI, port: int, host: str | None = None) -> None:
    """Run the app and print the URL that carries its access token.

    ``host`` defaults to ``$CAPX_WEB_UI_HOST``, else 127.0.0.1 (this machine only). The
    token is in the URL fragment, which the browser does not send to the server. Where
    the browser reaches the server at another address (a published container port), set
    ``$CAPX_WEB_UI_PUBLIC_URL``: a URL printed for the wrong port could reach someone
    else's server, which would read the token.
    """
    host = host or os.environ.get("CAPX_WEB_UI_HOST") or "127.0.0.1"
    # The literal address: "localhost" may resolve to ::1, where another user could listen.
    url_host = {"0.0.0.0": "127.0.0.1", "::": "[::1]"}.get(host, f"[{host}]" if ":" in host else host)
    base_url = os.environ.get("CAPX_WEB_UI_PUBLIC_URL") or f"http://{url_host}:{port}"
    print(f"\n  CaP-X Interactive Web UI: {base_url.rstrip('/')}/#token={app.state.access_token}\n")
    if host not in _LOOPBACK_HOSTS:
        logger.warning(
            f"Listening on {host}: anyone who can reach port {port} and has the token can run code as this user."
        )
    uvicorn.run(app, host=host, port=port)


# ============================================================================
# CLI Entry Point
# ============================================================================


@dataclass
class ServerArgs:
    """Command-line arguments for the web server."""

    host: str | None = None
    """Address to listen on. Default: $CAPX_WEB_UI_HOST, else 127.0.0.1 (this machine only)."""

    port: int = 8200
    """Port to run the server on."""


def main(args: ServerArgs | None = None) -> None:
    """Run the web server."""
    if args is None:
        args = tyro.cli(ServerArgs)
    serve(create_app(), port=args.port, host=args.host)


if __name__ == "__main__":
    main()
