"""HTTP over Unix domain sockets for the local model, perception and IK servers.

A local server is addressed as ``http+unix://<socket>/<path>``. ``<socket>`` is the
socket file's path, percent-encoded (``%2Fhome%2Fme%2Fsam3.sock``), or a bare name
such as ``sam3.sock`` that resolves inside the socket directory: ``$CAPX_SOCKET_DIR``,
by default ``~/.capx/sockets``. Plain ``http://`` is refused; ``https://`` stays for
remote APIs. Clients send through :mod:`capx.utils.unix_http_client`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from urllib.parse import quote, unquote, urlsplit

from pydantic import AfterValidator

from capx.protocol.framing import check_unix_path

if TYPE_CHECKING:
    from fastapi import FastAPI

SCHEME = "http+unix"
SOCKET_DIR_ENV = "CAPX_SOCKET_DIR"

logger = logging.getLogger(__name__)


def socket_dir() -> Path:
    raw = os.environ.get(SOCKET_DIR_ENV, "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".capx" / "sockets"


def resolve_socket(socket_name: str) -> Path:
    """The socket file for ``socket_name``: an absolute path as is, anything else
    inside :func:`socket_dir`."""
    if not socket_name:
        raise ValueError("empty Unix socket name")
    path = Path(socket_name).expanduser()
    if not path.is_absolute():
        path = socket_dir() / path
    check_unix_path(str(path))
    return path


def ensure_private_dir(directory: Path) -> Path:
    """Create ``directory`` as 0700, or check that an existing one is that private."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.stat()
    if info.st_uid != os.getuid():
        raise PermissionError(
            f"socket directory {directory} belongs to uid {info.st_uid}, not to you; "
            f"set {SOCKET_DIR_ENV} to a directory of your own"
        )
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"socket directory {directory} has mode {mode:o}, so other users can reach "
            f"the servers in it; run `chmod 700 {directory}`"
        )
    return directory


def unix_url(socket_name: str | os.PathLike[str], path: str = "") -> str:
    """The ``http+unix://`` URL of ``path`` on the server at ``socket_name``."""
    return f"{SCHEME}://{quote(os.fspath(socket_name), safe='')}{path}"


def check_server_url(url: str) -> str:
    """``url`` if a client may send to it: a local server's ``http+unix://`` URL, or
    a remote API's ``https://`` one."""
    parts = urlsplit(url)
    if parts.scheme == SCHEME and parts.netloc:
        resolve_socket(unquote(parts.netloc))
        return url
    if parts.scheme == "https" and parts.netloc:
        return url
    raise ValueError(
        f"server URL {url!r} is not allowed: local servers listen only on Unix sockets, "
        f"so address them as {SCHEME}://<socket>/<path> (for example "
        f"{SCHEME}://model.sock/v1/chat/completions, with model.sock in ${SOCKET_DIR_ENV}); "
        "a remote API needs https://"
    )


ServerUrl = Annotated[str, AfterValidator(check_server_url)]


def is_listening(path: Path, timeout_s: float = 1.0) -> bool:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout_s)
        try:
            probe.connect(str(path))
        except OSError:
            return False
    return True


def bind_unix_socket(socket_name: str) -> socket.socket:
    """A listening socket at ``socket_name``, mode 0600 in a 0700 directory.

    A socket file left by a server that is gone is replaced; a live one is not.
    """
    path = resolve_socket(socket_name)
    ensure_private_dir(path.parent)
    if path.exists() or path.is_symlink():
        if not stat.S_ISSOCK(path.lstat().st_mode):
            raise FileExistsError(f"{path} exists and is not a socket")
        if is_listening(path):
            raise OSError(f"another server is already listening on {path}")
        path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(path))
        os.chmod(path, 0o600)
        sock.listen(2048)
    except BaseException:
        sock.close()
        raise
    return sock


def serve_app(app: FastAPI, socket_name: str) -> None:
    """Run ``app`` with uvicorn on the Unix socket ``socket_name`` until stopped."""
    import uvicorn

    sock = bind_unix_socket(socket_name)
    path = sock.getsockname()
    try:
        logger.info("serving on unix socket %s", path)
        uvicorn.Server(uvicorn.Config(app)).run(sockets=[sock])
    finally:
        sock.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
