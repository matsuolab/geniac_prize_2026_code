# LIBERO-PRO タスク

[公式ドキュメント](https://github.com/capgym/cap-x/blob/main/docs/libero-tasks.md)


CaP-XはLIBERO-PROのタスクをサポートしています．

## セットアップ

コンテナ内でLIBERO専用の仮想環境を作成します（.venv-liberoが作成されます）．

```bash
bash /workspace/docker/setup-libero.sh
```


## タスクの実行


### GUIなしで実行

```bash
source .venv-libero/bin/activate
python capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model Qwen/Qwen3.6-27B \
    --server-url http+unix://model.sock/v1/chat/completions \
    --total-trials 10 --num-workers 2
```
`model`と`server-url`は使用するLLMに応じて変更してください．ローカルのサーバーはUnixソケットで待ち受けるので、URLは`http+unix://<ソケット>/<パス>`の形です（[README](../README.md#注意点)）．

### Web UI（現状ローカルLLMでは動作しない）

```bash
source .venv-libero/bin/activate
python capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model Qwen/Qwen3.6-27B \
    --server-url http+unix://model.sock/v1/chat/completions \
    --total-trials 10 --num-workers 2 --web-ui True
```

> **注記:** 知覚サーバー（SAM3，ContactGraspNet，PyRoKi）はYAMLが指定しているためタスク実行の際に自動起動します．

## タスクの選択

タスクの設定はYAMLファイルとして作成し，タスク実行時の`config-path`にファイルのパスを渡します．

各 LIBERO タスクは**スイート名**と**タスクID**で指定されます．YAMLの `low_level` フィールドは次のように書きます:

```
low_level:
  _target_: capx.envs.simulators.libero.FrankaLiberoEnv
  suite_name: <suite_name>
  task_id: <task_id>
```

例えば，`libero_goal` のタスク 2（「ワインボトルをキャビネットの上に置く」）を実行するには，次のように設定します:

```yaml
low_level:
  _target_: capx.envs.simulators.libero.FrankaLiberoEnv
  suite_name: libero_goal
  task_id: 2
```

##タスク設定の作成

既存のYAMLファイルをコピーして `suite_name` と `task_id` を変更します:

```bash
cp env_configs/libero/franka_libero_spatial_0.yaml env_configs/libero/franka_libero_goal_5.yaml
```

その後，新しいファイル内の `suite_name` と `task_id` を編集します:

```yaml
low_level:
  _target_: capx.envs.simulators.libero.FrankaLiberoEnv
  suite_name: libero_goal
  task_id: 5
```

### 利用可能なタスク一覧
#### LIBEROのタスク
| スイート | タスク数 | 説明 |
|-------|-------|-------------|
| `libero_10` | 10 | コアベンチマークタスク（複数オブジェクト・複数ステップ） |
| `libero_90` | 90 | 拡張ベンチマーク（多様なキッチン/リビングルームのシーン） |
| `libero_object` | 10 | オブジェクト汎化（同じシーン，異なるオブジェクト） |
| `libero_spatial` | 10 | 空間汎化（同じオブジェクト，異なる位置） |
| `libero_goal` | 10 | ゴール汎化（同じシーン，異なる） |

#### LIBERO-PROのタスク

| スイート | タスク数 | 説明 |
|-------|-------|-------------|
| `libero_object_swap` | 10 | `libero_object` に対する位置の摂動 |
| `libero_spatial_swap` | 10 | `libero_spatial` に対する位置の摂動 |
↓ここから下のタスクはダウンロードが必要
| `libero_10_swap` | 10 | `libero_10` に対する位置の摂動 |
| `libero_10_task` | 10 | `libero_10` に対するタスクの摂動 |
| `libero_object_task` | 10 | `libero_object` に対するタスクの摂動 |
| `libero_spatial_task` | 10 | `libero_spatial` に対するタスクの摂動 |
| `libero_goal_swap` | 10 | `libero_goal` に対する位置の摂動 |
| `libero_goal_task` | 10 | `libero_goal` に対するタスクの摂動 |

ダウンロードが必要なタスクは以下のコマンドによりhuggingfaceからダウンロードできます．

```bash
source /workspace/cap-x/.venv-libero/bin/activate
REPO=/workspace/cap-x/capx/third_party/LIBERO-PRO/libero/libero

# swap/task 全suiteを bddl+init まとめて1コマンドでDL
hf download zhouxueyang/LIBERO-Pro --repo-type dataset \
  --include "*_swap/*" "*_task/*" --local-dir /tmp/libero_pro

# 配置（既存 libero_object_swap / libero_spatial_swap は cp -n で保護）
cp -rn /tmp/libero_pro/bddl_files/* "$REPO/bddl_files/"
cp -rn /tmp/libero_pro/init_files/* "$REPO/init_files/"
```

その他のすべての LIBERO-PRO 摂動（Obj，Sem，Env）もサポートされています．正確なスイート名を含む完全な一覧を表示するには，ターミナルで次を実行してください．

```bash
source .venv-libero/bin/activate
python -c "
from libero import benchmark  # type: ignore[import-not-found]
benchmark_dict = benchmark.get_benchmark_dict(help=True)"
```

### タスクの例

**libero_10:**
| ID | タスク |
|-------|------|
| 0 | アルファベットスープとトマトソースの両方をバスケットに入れる |
| 1 | クリームチーズの箱とバターの両方をバスケットに入れる |
| 2 | コンロをつけてモカポットをその上に置く |

**libero_spatial:**
| ID | タスク |
|-------|------|
| 0 | 皿とラメキンの間にある黒いボウルを持ち上げて皿の上に置く |
| 1 | ラメキンの隣にある黒いボウルを持ち上げて皿の上に置く |
| 2 | テーブル中央にある黒いボウルを持ち上げて皿の上に置く |

**libero_goal:**
| ID | タスク |
|-------|------|
| 0 | キャビネットの真ん中の引き出しを開ける |
| 1 | ボウルをコンロの上に置く |
| 2 | ワインボトルをキャビネットの上に置く |

スイート内のすべてのタスクを一覧表示するには:

```bash
source .venv-libero/bin/activate
python -c "
from libero import benchmark
suite = benchmark.get_benchmark_dict()['libero_spatial']()
for i in range(suite.n_tasks):
    print(f'  [{i}] {suite.get_task(i).language}')
"
``` 

## 設定リファレンス

```yaml
env:
  _target_: capx.envs.tasks.franka.franka_libero_env.FrankaLiberoCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level:
      _target_: capx.envs.simulators.libero.FrankaLiberoEnv
      suite_name: libero_goal
      task_id: 5
    privileged: true                                       # true ：特権的なモード
    apis:
      - FrankaLiberoPrivilegedApi                          # 特権的なAPI
    prompt: |
      You are controlling a Fra・nka Emika robot with API described below.
      Goal: {libero_environment_goal}
      ...
```

タスクプロンプトは，`{libero_environment_goal}` プレースホルダーを通じて LIBERO のタスク言語記述から自動的に埋め込まれます．

## 利用可能な API
YAMLファイルの`apis`を変更することでLLMが用いることができるAPIを設定できます．
| API | 説明 |
|-----|-------------|
| `FrankaLiberoPrivilegedApi` | 直接物体の姿勢を獲得（知覚をスキップ），IK ベースの制御（Privileged） |
| `FrankaLiberoApi` | SAM3 + GraspNet による知覚ベースの制御（サーバーが必要） |
| `FrankaLiberoApiReduced` | 低レベルな知覚および制御機能（サーバーが必要） |
| `FrankaLiberoApiReducedSkillLibrary` | 低レベルな知覚および制御機能 + 自動合成されたスキルライブラリのヘルパ関数（サーバーが必要） |

## CuRobo の使用

まず、CuRoboを仮想環境にインストールします。

```bash
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet --extra curobo
```

CuRobo を使うには，使いたい API（すなわち `capx/integrations/franka/libero.py`，`capx/integrations/franka/libero_reduced.py`）内の以下の関数のコメントアウトを解除してください．

```python
def functions(self) -> dict[str, Any]:
  fns =  {
      ...
      # # CuRobo, uncomment these for the coding agent to use them!
      # "plan_grasp_trajectory": self.plan_grasp_trajectory,
      # "plan_with_grasped_object": self.plan_with_grasped_object,
      # "execute_joint_trajectory": self.execute_joint_trajectory,
  }
```

また、CuroboのAPIの使用には別でサーバーを起動する必要があります．

YAMLファイルの`api_servers`に以下のようにCuroboのサーバーを追加してください。

```yaml
api_servers:
  - _target_: capx.serving.launch_pyroki_server.main
    socket: pyroki.sock
    robot: panda_description
    target_link: panda_hand
  # ↓ 追加
  - _target_: capx.serving.launch_curobo_server.main
    socket: curobo.sock
```
