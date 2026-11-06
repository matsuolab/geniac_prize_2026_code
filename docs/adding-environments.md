# 新しいシミュレータ・タスク・ロボットの追加

[公式ドキュメント](https://github.com/capgym/cap-x/blob/main/docs/adding-environments.md)

## 目的

CaP-X は、LLM が生成した Python コードでロボットを操作させ、その成否（reward・動画・実行ログなど）を評価するための共通基盤（`capx/envs/launch.py` によるプロンプト生成 → コード実行 → 集計の一連の流れ、並列実行、Web UI など）を提供しています。

このドキュメントの目的は、その共通基盤に載せる形で

- 新しいシミュレータ（独自の物理エンジンや実機など）
- 新しいロボット（既存の Franka / R1Pro 以外のロボット）
- 新しいタスク（独自の把持・組み立てタスクなど）

を追加する手順を示すことです。既存の `franka_*` / `r1pro_*` タスクと同じインターフェース（`BaseEnv` / `CodeExecutionEnvBase` / `ApiBase`）に従うことで、トライアル実行・報酬集計・動画記録・並列ワーカー・Web UI といった機能をそのまま使い回せます。

以下の Step 1〜4 は、実際に最小構成のダミーシミュレータとタスクを作成し、`uv run --no-sync --active capx/envs/launch.py --config-path ... --use-oracle-code True` で最後まで動作することを確認した内容を反映しています。

## Step 1: シミュレータのラッパーを作成する

`capx/envs/simulators/` 内で `BaseEnv` をサブクラス化します。以下の 5 つの必須メソッドを実装してください。

```python
# capx/envs/simulators/my_sim.py
from capx.envs.base import BaseEnv

class MySimEnv(BaseEnv):
    def __init__(self, max_steps=1000, privileged=False, enable_render=False, viser_debug=False):
        super().__init__()
        self.max_steps = max_steps
        # Initialize your simulator here

    def reset(self, *, seed=None, options=None):
        # Reset simulator, return (observation_dict, info_dict)
        ...

    def step(self, action):
        # Execute action, return (obs, reward, terminated, truncated, info)
        ...

    def get_observation(self):
        # Return dict with camera images, robot state, object poses
        # Required keys for vision APIs:
        #   obs["robot0_robotview"]["images"]["rgb"]   — (H, W, 3) uint8
        #   obs["robot0_robotview"]["images"]["depth"] — (H, W, 1) float32
        #   obs["robot0_robotview"]["intrinsics"]      — (3, 3) float64
        #   obs["robot0_robotview"]["pose_mat"]        — (4, 4) float64
        ...

    def compute_reward(self):
        # Return float reward for the current state
        ...

    def task_completed(self):
        # Return bool — has the task been solved?
        ...
```

> **注意:** `capx/envs/base.py` の `get_env()` は環境を生成する際に常に `viser_debug` キーワード引数を渡します。`__init__` に `viser_debug=False` を含めておかないと、環境取得時に `TypeError: unexpected keyword argument 'viser_debug'` になります。

`capx/envs/simulators/__init__.py` に登録します。

```python
from capx.envs.base import register_env
from .my_sim import MySimEnv
register_env("my_sim_env", MySimEnv)
```

### `launch.py` でトライアルを実行するために追加で必要なメソッド

上記の 5 メソッドは `BaseEnv` が要求する最小限の抽象メソッドですが、`capx/envs/launch.py`（Step 4）経由でトライアルを実行し、YAML 設定で `record_video: true` にする場合は、`capx/envs/trial.py` が動画バッファリングのために以下のメソッドも呼び出します。これらが無いと `AttributeError: 'MySimEnv' object has no attribute 'enable_video_capture'` でトライアルが失敗します。

```python
    def enable_video_capture(self, enabled=True, *, clear=True, wrist_camera=False):
        # Start/stop buffering rendered frames
        ...

    def get_video_frames(self, *, clear=False):
        # Return buffered frames as a list of (H, W, 3) uint8 arrays
        ...
```

（`get_video_frame_count()` / `get_video_frames_range()` / `render_wrist()` は呼び出し側で `hasattr` チェックされるため実装は任意です。）

> **Tip:** Robosuite ベースの環境を作る場合は、`robosuite_base.py` の `RobosuiteBaseEnv` を継承すると、`viser_debug` 対応、動画キャプチャ（`enable_video_capture` / `get_video_frames` など）、カメラ処理、モーションコントロールを無償で利用できます。

## Step 2: タスク環境を作成する

各タスクは、プロンプトとオラクルコードを定義する薄いラッパーです。

```python
# capx/envs/tasks/my_robot/my_task.py
from capx.envs.tasks.base import CodeExecutionEnvBase

class MyTaskCodeEnv(CodeExecutionEnvBase):
    """Short description of the task."""

    prompt = """
    You are controlling a robot with the API described below.
    Goal: [describe the task clearly]

    Key rules:
    - [important constraints the model should follow]
    - [coordinate frame conventions, units, etc.]

    Write ONLY executable Python code. No code fences.
    """

    oracle_code = """
    import numpy as np
    # Reference solution that achieves the task
    home_pose()
    pos, quat = sample_grasp_pose("target object")
    ...
    """
```

`capx/envs/tasks/__init__.py` に登録します。

```python
from capx.envs.tasks.base import CodeExecEnvConfig, register_exec_env, register_config
from .my_robot.my_task import MyTaskCodeEnv

register_exec_env("my_task_code_env", MyTaskCodeEnv)
register_config("my_task_code_env", CodeExecEnvConfig(
    low_level="my_sim_env",     # Name registered in Step 1
    apis=["FrankaControlApi"],  # Which APIs to expose
))
```

設定（config）と環境の登録を init ファイルに定義しているのは、環境とそれに対応する設定ファイルを見つけるための統一されたアクセスポイントを一つにまとめたいためです。

## Step 3: YAML 設定を作成する

```yaml
# env_configs/my_task/my_task.yaml
env:
  _target_: capx.envs.tasks.my_robot.my_task.MyTaskCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: my_sim_env
    privileged: false
    apis:
      - FrankaControlApi

record_video: true
output_dir: ./outputs/my_task
trials: 100
num_workers: 12
```

## Step 4: テストする

```bash
# Run oracle code to verify the environment works
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/my_task/my_task.yaml \
    --use-oracle-code True --total-trials 5

# Run with an LLM
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/my_task/my_task.yaml --total-trials 20
```

## 新しい B1K タスクの追加

B1K/BEHAVIOR タスクは、`capx/envs/simulators/r1pro_b1k.py` の `R1ProBehaviourLowLevel` が OmniGibson の `activity_name` を切り替えることで実現されています。`controller_cfg` はタスクごとに新規作成するファイルではなく、`og.example_config_path`（OmniGibson 側が持つ既定のディレクトリ）から読み込まれる固定のプリミティブ定義 `r1pro_primitives.yaml` です。

1. `env_configs/r1pro` の下に新しい設定ファイルを作成し、`low_level` に `activity_name: ${TASK_NAME}`（OmniGibson/BEHAVIOR の activity 名）を指定します（`controller_cfg` は既定の `r1pro_primitives.yaml` のままで構いません）。

```yaml
# env_configs/r1pro/${TASK_NAME}.yaml
env:
  _target_: capx.envs.tasks.r1pro.r1pro_behavior.R1ProBehaviorCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level:
      _target_: capx.envs.simulators.r1pro_b1k.R1ProBehaviourLowLevel
      controller_cfg: r1pro_primitives.yaml
      activity_name: ${TASK_NAME}
    privileged: false
    apis:
      - R1ProControlApi
```

2. 実行します。
```bash
uv run --no-sync --active capx/envs/launch.py --config-path env_configs/r1pro/${TASK_NAME}.yaml
```

## 利用可能な環境の一覧表示

```python
from capx.envs.tasks import list_exec_envs, list_configs
from capx.envs import list_envs
from capx.integrations import list_apis

print("Available environments: ", list_envs())
print("Available execution environments: ", list_exec_envs())
print("Available APIs: ", list_apis())
print("Available configurations: ", list_configs())
```
