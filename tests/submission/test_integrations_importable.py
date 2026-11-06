"""Importability guard for ``capx/integrations/r1pro/{control,utils}.py``.

These two modules must be
importable in a sim-free client venv (no omnigibson/BEHAVIOR) with zero
behavioral change when the sim stack IS installed. Two complementary layers:

1. A static AST check (always runs, no dependency on what happens to be
   installed) that fails hard if either module has an ``omnigibson`` or
   ``capx.envs.simulators.r1pro_b1k`` import that executes at module-import
   time (i.e. not gated by ``if TYPE_CHECKING:`` and not inside a
   function/method body).
2. Subprocess dynamic-import checks (``python -c`` via ``sys.executable``,
   per the design's test plan) that actually import the modules. These MUST
   fail if the sim stack leaks in at import time, and MAY skip (with reason)
   when the only obstacle is a missing third-party client dependency that is
   genuinely unrelated to the sim-import guard (e.g. jax/pyroki, which
   R1ProControlApi legitimately needs for its in-process IK solver but which
   this repo's sim-free test venv intentionally excludes).
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
R1PRO_DIR = REPO_ROOT / "capx" / "integrations" / "r1pro"

# Modules that must never appear as the (proximate) cause of an import
# failure -- their presence means the sim stack
# leaked into a module-scope import.
_FORBIDDEN_ROOT_MODULES = ("omnigibson", "capx.envs.simulators.r1pro_b1k")


def _is_type_checking_test(test: ast.expr) -> bool:
    """Return True if an ``if`` test is (roughly) ``typing.TYPE_CHECKING``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _module_scope_imported_names(source: str) -> set[str]:
    """Collect the module names touched by imports that run at import time.

    Walks the statement tree but does NOT descend into function/method
    bodies (never executed at import time) and treats the body of
    ``if TYPE_CHECKING:`` as dead code (False at runtime), matching how
    CPython actually executes a module.
    """
    tree = ast.parse(source)
    names: set[str] = set()

    def walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # bodies only run when called, not at import time
            if isinstance(stmt, ast.Import):
                names.update(alias.name for alias in stmt.names)
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.module:
                    # Track the fully qualified module a `from X import Y`
                    # pulls, e.g. "capx.envs.simulators.r1pro_b1k".
                    names.add(stmt.module)
            elif isinstance(stmt, ast.If):
                if _is_type_checking_test(stmt.test):
                    walk(stmt.orelse)  # only the else-branch runs for real
                else:
                    walk(stmt.body)
                    walk(stmt.orelse)
            elif isinstance(stmt, ast.ClassDef):
                walk(stmt.body)  # class bodies DO execute at import time
            elif isinstance(stmt, ast.Try):
                walk(stmt.body)
                for handler in stmt.handlers:
                    walk(handler.body)
                walk(stmt.orelse)
                walk(stmt.finalbody)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                walk(stmt.body)
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                walk(stmt.body)
                walk(stmt.orelse)

    walk(tree.body)
    return names


def _assert_no_forbidden_module_scope_import(path: Path) -> None:
    names = _module_scope_imported_names(path.read_text())
    offenders = {
        n
        for n in names
        if any(n == forbidden or n.startswith(forbidden + ".") for forbidden in _FORBIDDEN_ROOT_MODULES)
    }
    assert not offenders, (
        f"{path} imports {sorted(offenders)} at module scope; these must be "
        "under `if TYPE_CHECKING:` or a lazy (in-function) import so the "
        "module stays importable sim-free."
    )


def test_control_py_has_no_module_scope_sim_imports() -> None:
    _assert_no_forbidden_module_scope_import(R1PRO_DIR / "control.py")


def test_utils_py_has_no_module_scope_sim_imports() -> None:
    _assert_no_forbidden_module_scope_import(R1PRO_DIR / "utils.py")


def test_control_py_keeps_r1pro_behaviour_low_level_as_annotation_only() -> None:
    """The only reference to R1ProBehaviourLowLevel must be under TYPE_CHECKING."""
    source = (R1PRO_DIR / "control.py").read_text()
    tree = ast.parse(source)
    top_level_names = _module_scope_imported_names(source)
    assert "capx.envs.simulators.r1pro_b1k" not in top_level_names

    found_in_type_checking_guard = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom) and stmt.module == "capx.envs.simulators.r1pro_b1k":
                    if any(alias.name == "R1ProBehaviourLowLevel" for alias in stmt.names):
                        found_in_type_checking_guard = True
    assert found_in_type_checking_guard, (
        "expected `from capx.envs.simulators.r1pro_b1k import R1ProBehaviourLowLevel` "
        "under `if TYPE_CHECKING:` in control.py"
    )
    # `from __future__ import annotations` is required so the (now string)
    # `env: R1ProBehaviourLowLevel` parameter annotation is never evaluated
    # at runtime -- otherwise the TYPE_CHECKING guard alone would still
    # raise NameError at class-definition (i.e. import) time.
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__" and any(a.name == "annotations" for a in node.names)
        for node in tree.body
    ), "control.py must start with `from __future__ import annotations`"


def _run_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _classify_failure(proc: subprocess.CompletedProcess[str]) -> str:
    """Return the module name from the last ModuleNotFoundError in stderr.

    Raises AssertionError (failing the calling test) if the failure is
    unrelated to a missing module (an unexpected error), and returns the
    empty string if there was no failure at all.
    """
    if proc.returncode == 0:
        return ""
    matches = re.findall(r"ModuleNotFoundError: No module named '([^']+)'", proc.stderr)
    if not matches:
        pytest.fail(f"unexpected non-import failure (rc={proc.returncode}):\n{proc.stderr}")
    return matches[-1]


def _assert_not_sim_leak_or_skip(missing_module: str, stderr: str) -> None:
    if any(missing_module == forbidden or missing_module.startswith(forbidden + ".") for forbidden in _FORBIDDEN_ROOT_MODULES):
        pytest.fail(
            "capx.integrations.r1pro.control pulled in the sim stack "
            f"({missing_module!r}) at import time:\n{stderr}"
        )
    pytest.skip(
        f"missing third-party client dependency {missing_module!r}, unrelated to "
        f"the r1pro sim-import guard (this venv intentionally excludes it):\n"
        f"{stderr[-2000:]}"
    )


def test_r1pro_control_importable_isolated_from_sibling_apis() -> None:
    """Import capx.integrations.r1pro.control without capx/integrations/__init__.py.

    capx/integrations/__init__.py unconditionally imports sibling API modules
    (e.g. capx.integrations.franka.control) before ever reaching r1pro; those
    modules have their own third-party requirements (pyroki/jax) that are
    unrelated to the sim-import guard and are intentionally absent from this test venv. Stub
    the parent package in sys.modules so this test isolates exactly the
    r1pro submodule's own import chain.
    """
    snippet = (
        "import sys, types, os\n"
        "import capx\n"
        "pkg = types.ModuleType('capx.integrations')\n"
        "pkg.__path__ = [os.path.join(os.path.dirname(capx.__file__), 'integrations')]\n"
        "sys.modules['capx.integrations'] = pkg\n"
        "from capx.integrations.r1pro.control import R1ProControlApi\n"
        "assert 'omnigibson' not in sys.modules\n"
        "assert 'capx.envs.simulators.r1pro_b1k' not in sys.modules\n"
        "print('ISOLATED_IMPORT_OK', R1ProControlApi)\n"
    )
    proc = _run_snippet(snippet)
    missing = _classify_failure(proc)
    if missing:
        _assert_not_sim_leak_or_skip(missing, proc.stderr)
        return
    assert "ISOLATED_IMPORT_OK" in proc.stdout, proc.stdout + proc.stderr


def test_capx_integrations_registers_r1pro_control_api_or_skips() -> None:
    """`import capx.integrations` registers the R1Pro API.

    Runs through the REAL package __init__.py (unlike the isolated test
    above), so this can only fully pass in a venv with every client-side
    dependency capx.integrations pulls in (jax/pyroki included). It still
    enforces the hard requirement: any failure caused by the sim stack
    is a test failure, not a skip.
    """
    snippet = (
        "import capx.integrations as m\n"
        "apis = m.list_apis()\n"
        "assert 'FrankaControlApi' in apis, sorted(apis)\n"
        "assert 'R1ProControlApi' in apis, sorted(apis)\n"
        "print('REGISTERED_OK')\n"
    )
    proc = _run_snippet(snippet)
    missing = _classify_failure(proc)
    if missing:
        _assert_not_sim_leak_or_skip(missing, proc.stderr)
        return
    assert "REGISTERED_OK" in proc.stdout, proc.stdout + proc.stderr


def test_capx_envs_tasks_base_importable_or_skips() -> None:
    """Guard the capx.envs.tasks.base import chain too.

    (GatewayCodeExecEnv subclasses CodeExecutionEnvBase, which transitively
    imports capx.integrations.__init__ -- this pins that the ONLY reason it
    can fail sim-free is a missing third-party client dependency, never the
    sim stack.)
    """
    snippet = "import capx.envs.tasks.base\nprint('TASKS_BASE_IMPORT_OK')\n"
    proc = _run_snippet(snippet)
    missing = _classify_failure(proc)
    if missing:
        _assert_not_sim_leak_or_skip(missing, proc.stderr)
        return
    assert "TASKS_BASE_IMPORT_OK" in proc.stdout, proc.stdout + proc.stderr


def test_relative_pose_transform_matches_omnigibson() -> None:
    """Numeric parity of the ported relative_pose_transform.

    Skipped when omnigibson isn't installed (the common case for this
    sim-free venv); runs for real in the BEHAVIOR container.
    """
    omnigibson_transform_utils = pytest.importorskip("omnigibson.utils.transform_utils")
    import torch

    from capx.integrations.r1pro.utils import relative_pose_transform

    rng = __import__("numpy").random.default_rng(0)
    for _ in range(5):
        pos1 = torch.tensor(rng.normal(size=3), dtype=torch.float32)
        pos0 = torch.tensor(rng.normal(size=3), dtype=torch.float32)
        quat1 = torch.nn.functional.normalize(torch.tensor(rng.normal(size=4), dtype=torch.float32), dim=0)
        quat0 = torch.nn.functional.normalize(torch.tensor(rng.normal(size=4), dtype=torch.float32), dim=0)

        got_pos, got_quat = relative_pose_transform(pos1, quat1, pos0, quat0)
        want_pos, want_quat = omnigibson_transform_utils.relative_pose_transform(pos1, quat1, pos0, quat0)

        torch.testing.assert_close(got_pos, want_pos, atol=1e-5, rtol=1e-5)
        # Quaternions may differ by sign (q and -q are the same rotation).
        sign = 1.0 if torch.dot(got_quat, want_quat) >= 0 else -1.0
        torch.testing.assert_close(got_quat, sign * want_quat, atol=1e-5, rtol=1e-5)
