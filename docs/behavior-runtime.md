# BEHAVIOR backend lifecycle

The persistent backend starts a fresh worker process for each environment. Isaac Sim imports,
scene creation and simulator calls run on that worker's process main thread. Closing an episode
terminates and waits for its owned worker before another environment is built. Other simulators
keep the existing threaded backend dispatch.

Scene lookup uses the requested public activity's installed template. Reward and success read
the native task result from the last step, without advancing physics or enumerating legacy demo
metrics. Camera startup waits for an actual valid intrinsic matrix with a bounded render warmup.

Each lane defaults to one CPU worker, with a minimum of four Kit launcher threads. Override the
worker budget with `CAPX_LANE_CPUS` before starting `docker/serve_env_backend.py`. Numeric-library
thread limits are applied before their imports. Initialization remains serialized, so more lanes
can still increase reset waiting time and action latency.

Use an owned writable cache and work directory. `CAPX_WORKER_ROOT` defaults to `/work/w` and must
be short enough for a Unix socket. `CAPX_INIT_LOCK` defaults to
`/cache/behavior-initialization.lock`; lanes sharing a GPU must see the same lock file, including
through a shared bind mount when they run in separate containers. Keep cache/work directories
outside source and dataset mounts.

CPU regressions cover delayed camera calibration, main-thread dispatch, scene lookup, native
scoring, repeated environment creation and worker-crash recovery. They use synthetic state and
the public task pack; no private task pack or simulator installation is required.
