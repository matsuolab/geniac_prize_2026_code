"""Short, private socket paths for tests: pytest's tmp_path can exceed the AF_UNIX limit."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile


def fresh_socket_path(name: str = "s.sock") -> str:
    directory = tempfile.mkdtemp(prefix="capx-test-")
    atexit.register(shutil.rmtree, directory, True)
    return os.path.join(directory, name)
