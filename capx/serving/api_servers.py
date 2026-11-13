from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from capx.utils.unix_http import resolve_socket


class ApiServerSpec(BaseModel):
    """One ``api_servers`` entry: the server's ``main``, the Unix socket it serves on,
    and any other keyword arguments for ``main``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    target: str = Field(alias="_target_")
    socket: str

    @model_validator(mode="before")
    @classmethod
    def _reject_tcp(cls, data: object) -> object:
        if isinstance(data, Mapping):
            tcp = sorted({"host", "port"} & set(data))
            if tcp:
                raise ValueError(
                    f"api_servers entry {data.get('_target_')!r} sets {' and '.join(tcp)}; "
                    "servers listen only on Unix sockets, so replace them with "
                    "`socket: <name>.sock`"
                )
        return data

    @field_validator("socket")
    @classmethod
    def _fits(cls, socket: str) -> str:
        resolve_socket(socket)
        return socket
