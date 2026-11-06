# CaP-X Interactive Web UI

A modern chat-based interface for real-time robot code execution demos.

## Features

- **Chat Interface**: Interactive chat view showing model responses, code blocks, and execution results
- **Visual Feedback**: Display environment images captured during execution
- **Thinking Traces**: Collapsible sections for model reasoning
- **User Injection**: Inject custom prompts during multi-turn execution
- **3D Visualization**: Embedded iframe for live robot visualization (Viser)
- **Real-time Updates**: WebSocket streaming for instant feedback

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser (React + Vite)                      │
│  ┌─────────────────────────────┬───────────────────────────────────┐│
│  │      Chat Panel (50%)       │    3D Visualization (50%)         ││
│  └─────────────────────────────┴───────────────────────────────────┘│
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ WebSocket
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (port 8200)                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites

- Python 3.10+ with `uv` package manager
- Node.js 18+ with npm

### Backend Setup

The backend is part of the CaP-X package. No additional setup needed.

### Frontend Setup

```bash
cd web-ui
npm install
```

> **After editing frontend code** (e.g. `src/App.tsx`), rebuild the production bundle before launching:
> ```bash
> cd web-ui && npm run build
> ```
> The `--web-ui True` flag in `launch.py` serves the built assets from `web-ui/dist/`. Without rebuilding, your changes won't take effect.

## Running

### 1. Start the Backend Server

```bash
# From the CaP-X root directory
uv run python -m capx.web.server --port 8200
```

The server prints a URL with an access token, `http://127.0.0.1:8200/#token=...`. Every
request except the built frontend's files needs that token. Opening the URL once keeps it
in the browser's storage for that origin (scheme, host and port; it is never put in a
cookie, because cookies also reach every other port on the host). Scripts send it as
`Authorization: Bearer <token>`. A new token is made at each start; with an old one the
page asks for the new URL.

The server listens on 127.0.0.1 only. From another machine, forward the port
(`ssh -L 127.0.0.1:8200:127.0.0.1:8200 <host>`). Use `127.0.0.1`, not `localhost`, in
these URLs: `localhost` can resolve to `::1`, where another user could be listening and
would receive the token. To listen on other addresses, pass `--host 0.0.0.0` (or set
`CAPX_WEB_UI_HOST=0.0.0.0` for `launch.py --web-ui True`); anyone who can reach the port
and has the token can then run code as you.

### 2. Start the Frontend Dev Server

```bash
cd web-ui
npm run dev
```

### 3. Open the UI

Open http://127.0.0.1:5173/#token=... with the token from the URL the backend printed.
The dev server is a different origin from the backend, so it needs the token itself, and
it keeps it in memory only (another user could take port 5173 once the dev server stops):
open the token URL again after reloading the page. The dev server refuses to start if
port 5173 is taken.

The 3D view shows the Viser server that the running trial's environment started, through
the backend's proxy. The proxied page runs sandboxed, in an origin of its own, so it cannot
read the token.

## Usage

1. **Load Config**: Select a YAML config file from the dropdown. Only configs under `env_configs/` (and the one given to `launch.py`) are accepted
2. **Configure Model**: Select the model and adjust server URL if needed
3. **Start Trial**: Click "Start Trial" to begin execution
4. **Monitor**: Watch the chat for model responses, code execution, and visual feedback
5. **Interact**: When "Pause each turn" is enabled, inject prompts during execution
6. **Stop**: Click "Stop" to abort at any time

## Development

### Frontend Structure

```
src/
├── components/
│   ├── ChatPanel.tsx       # Main chat container
│   ├── ChatMessage.tsx     # Individual message rendering
│   ├── CodeBlock.tsx       # Syntax-highlighted code
│   ├── ThinkingSection.tsx # Collapsible reasoning
│   ├── ImageViewer.tsx     # Visual feedback display
│   └── ...
├── hooks/
│   ├── useWebSocket.ts     # WebSocket connection
│   └── useTrialState.ts    # Trial state machine
└── types/
    └── messages.ts         # TypeScript types
```

### Backend Structure

```
capx/web/
├── server.py              # FastAPI app + WebSocket
├── models.py              # Pydantic schemas
├── session_manager.py     # Session tracking
└── async_trial_runner.py  # Async trial execution
```

## WebSocket Protocol

### Server → Client Events

| Event | Description |
|-------|-------------|
| `model_thinking` | Model is generating |
| `model_response` | Model response with code |
| `code_execution_start` | Code block starting |
| `code_execution_result` | Execution completed |
| `visual_feedback` | Environment image |
| `user_prompt_request` | Awaiting user input |
| `trial_complete` | Trial finished |
| `state_update` | State changed |
| `error` | Error occurred |

### Client → Server Commands

| Command | Description |
|---------|-------------|
| `inject_prompt` | Add user text to prompt |
| `stop` | Stop the trial |
| `resume` | Continue without injection |
