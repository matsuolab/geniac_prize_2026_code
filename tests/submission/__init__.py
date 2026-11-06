"""Unit tests for ``capx/submission/`` (the gateway migration client).

No simulator, no network: every test in this package drives the shims/runner
against ``tests.submission.fakes.FakeEnvClient`` instead of a real gateway.
"""
