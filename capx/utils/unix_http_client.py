"""A requests session that sends ``http+unix://`` URLs (see :mod:`capx.utils.unix_http`)
over the Unix socket they name."""

from __future__ import annotations

import socket
import threading
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.exceptions import NewConnectionError

from capx.utils.unix_http import SCHEME, check_server_url, resolve_socket


class _UnixConnection(HTTPConnection):
    def __init__(self, *args: Any, socket_path: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.socket_path = socket_path

    def _new_conn(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if isinstance(self.timeout, (int, float)):
            sock.settimeout(self.timeout)
        try:
            sock.connect(self.socket_path)
        except OSError as exc:
            sock.close()
            raise NewConnectionError(
                self, f"cannot connect to unix socket {self.socket_path}: {exc}"
            ) from exc
        return sock


class _UnixConnectionPool(HTTPConnectionPool):
    # urllib3's own HTTPConnection fails pyright's check against this protocol too.
    ConnectionCls = _UnixConnection  # pyright: ignore[reportAssignmentType]


class UnixSocketAdapter(HTTPAdapter):
    """Sends each ``http+unix://`` request to the socket its URL names, never via a proxy."""

    def __init__(self) -> None:
        super().__init__()
        self._unix_pools: dict[str, _UnixConnectionPool] = {}
        self._unix_lock = threading.Lock()

    def _unix_pool(self, url: str | None) -> _UnixConnectionPool:
        if url is None:
            raise ValueError("request has no URL")
        path = str(resolve_socket(unquote(urlsplit(url).netloc)))
        with self._unix_lock:
            pool = self._unix_pools.get(path)
            if pool is None:
                # Every lane shares one pool per server; a small one would drop
                # connections, with a warning each, under the ensemble's fan-out.
                pool = _UnixConnectionPool("localhost", maxsize=256, socket_path=path)
                self._unix_pools[path] = pool
            return pool

    def get_connection_with_tls_context(
        self,
        request: requests.PreparedRequest,
        verify: bool | str | None,
        proxies: Mapping[str, str] | None = None,
        cert: tuple[str, str] | str | None = None,
    ) -> HTTPConnectionPool:
        return self._unix_pool(request.url)

    def request_url(
        self, request: requests.PreparedRequest, proxies: Mapping[str, str] | None
    ) -> str:
        return request.path_url

    def close(self) -> None:
        super().close()
        with self._unix_lock:
            for pool in self._unix_pools.values():
                pool.close()
            self._unix_pools.clear()


class _CheckedRedirectSession(requests.Session):
    """Follows a redirect only to a URL :func:`check_server_url` allows, so a server
    cannot send a request body on over plain TCP."""

    def get_redirect_target(self, resp: requests.Response) -> str | None:
        target = super().get_redirect_target(resp)
        if target is None:
            return None
        base = urlsplit(resp.url)
        location = urlsplit(target)
        # A server behind a socket sees Host "localhost", so its own redirects
        # (FastAPI's trailing-slash one included) come back as http://localhost/...
        # or as a bare path; urljoin cannot resolve either against http+unix.
        if base.scheme == SCHEME and (
            not location.netloc or (location.scheme == "http" and location.hostname == "localhost")
        ):
            path = urljoin(base.path, location.path)
            target = urlunsplit((SCHEME, base.netloc, path, location.query, ""))
        return check_server_url(urljoin(resp.url, target))


def new_session() -> requests.Session:
    session = _CheckedRedirectSession()
    session.mount(f"{SCHEME}://", UnixSocketAdapter())
    return session


_shared: requests.Session | None = None
_shared_lock = threading.Lock()


def shared_session() -> requests.Session:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = new_session()
        return _shared


def post(
    url: str,
    *,
    json: object | None = None,
    data: str | bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | tuple[float, float] | None,
    stream: bool = False,
) -> requests.Response:
    """POST to a local server over its Unix socket, or to a remote API over https."""
    return shared_session().post(
        check_server_url(url),
        json=json,
        data=data,
        headers=headers,
        timeout=timeout,
        stream=stream,
    )


def get(url: str, *, timeout: float | tuple[float, float] | None) -> requests.Response:
    return shared_session().get(check_server_url(url), timeout=timeout)
