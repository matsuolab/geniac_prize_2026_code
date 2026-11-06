from capx.envs.tasks.base import CodeExecutionEnvBase


class FrankaLiberoCodeEnv(CodeExecutionEnvBase):
    """Generic high-level code environment for any LIBERO task.

    It does not hardcode a prompt or oracle code: both come from the YAML config
    (``CodeExecEnvConfig.prompt`` / ``oracle_code``).

    The LIBERO suite_name and task_id are specified on the low-level env
    (``FrankaLiberoEnv``) in the YAML config, so no per-task Python file is needed.
    """


__all__ = ["FrankaLiberoCodeEnv"]
