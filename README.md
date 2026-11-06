# GENIAC-PRIZE 2026 テーマ2 評価コード

[GENIAC-PRIZE 2026 テーマ2](https://geniac-prize.nedo.go.jp/theme2/) のコンペティションのためのリポジトリです。
評価環境、提出の仕組み、参照実装が入っています。

**予選のルールは [PRELIM_RULEBOOK.md](PRELIM_RULEBOOK.md) にすべて書いてあります。**

このリポジトリは、ロボットマニピュレーション向けのコーディングエージェントのフレームワーク
[CaP-X](https://github.com/capgym/cap-x)（[論文](https://arxiv.org/abs/2603.22435)、[プロジェクトページ](https://capgym.github.io/)）を元に、
コンペティション用の評価環境を加えたものです。CaP-X は次のものからできています。

| コンポーネント | 内容 |
| -------------- | ---- |
| **CaP-Gym**    | エージェントが、知覚と制御のプリミティブを組み合わせた Python コードを生成してロボットを操作する、対話的な Gymnasium 環境です。Robosuite、LIBERO-PRO、BEHAVIOR のタスクを収録しています。 |
| **CaP-Bench**  | コーディングエージェントを抽象化レベル・対話モード・視覚的グラウンディングの手法ごとに評価するベンチマークです。単一ターンの S1〜S4、複数ターンの M1〜M4 の8段階があります。 |
| **CaP-Agent0** | 複数ターンの視覚差分、自動合成したスキルライブラリ、並列アンサンブル推論を備えた、学習不要のエージェントです。コンペティションの参照実装でもあります。 |
| **CaP-RL**     | GRPO でコーディングエージェントを強化学習するモジュールです。 |

---

## コンペティションの評価

ルール、評価のしくみ、提出物の形式、インターフェースの仕様は [PRELIM_RULEBOOK.md](PRELIM_RULEBOOK.md) にあります。

- [提出の手順](docs/submission.md): 手元での評価、イメージの作成、HuggingFace への提出
- [評価環境の運用](docs/operations.md): 評価キューを動かす主催者向け

### すぐに試す

```bash
git clone --recurse-submodules https://github.com/matsuolab/geniac_prize_2026_code
cd geniac_prize_2026_code
```

`private/` は運営の非公開リポジトリの submodule で、clone では取得されません。

```bash
# シミュレータ不要のテスト
PYTHONPATH=. python -m pytest -q tests/test_protocol_runtime.py tests/test_gateway.py tests/test_config.py

# Robosuite を gateway と同じプロセスで動かして、公開パックの Robosuite タスクを評価する
PYTHONPATH=. python -m capx.cli.eval --taskpack taskpacks/original --mode dev --simulators robosuite
```

ホストにシミュレータを入れる手順は [docs/native-setup.md](docs/native-setup.md) にあります。本番と同じように gateway とシミュレータを別のコンテナで動かすには、Docker Compose を使います（[docker/README.md](docker/README.md)）。

```bash
docker compose up --build
```

シミュレータを別プロセスにして、CLI から指定することもできます。

```bash
# ターミナル1: シミュレータのバックエンド
CAPX_SIMULATOR=robosuite CAPX_BACKEND_PORT=8100 python docker/serve_env_backend.py

# ターミナル2: gateway と評価ループ（シミュレータは import しない）
python -m capx.cli.eval --taskpack taskpacks/original --mode dev \
    --simulators robosuite --backend robosuite=127.0.0.1:8100
```

複数のシミュレータを使うときは、`--backend robosuite=127.0.0.1:8100 libero=127.0.0.1:8200` のように1つの `--backend` の後に並べます。LIBERO のバックエンドは専用の venv（`.venv-libero`）で動かします。BEHAVIOR は [docs/behavior-tasks.md](docs/behavior-tasks.md) を見てください。

### 参照実装（CaP-Agent0）を動かす

CaP-Agent0（LLM の複数ターンループ、SAM3 と GraspNet による知覚、制御プリミティブ）は、`EnvClient` を通して3つのシミュレータすべてで動きます。

```bash
PYTHONPATH=. python -m capx.submission.run --config-path env_configs/gateway/runner.yaml
```

gateway には `CAPX_ENV_SOCKET` のソケット、なければ `CAPX_ENV_HOST`・`CAPX_ENV_PORT`（既定は `gateway:8000`）で接続します。タスクごとの設定（プロンプト、使う API）は `env_configs/gateway/*.yaml` にあり、`task_map` で選ばれます。必要なものは torch（CPU で可）、jax と pyroki、そして設定が参照するサーバー（vLLM、SAM3、GraspNet、PyRoKi、両腕のタスクでは OWL-ViT と SAM2）です。

### 自分のポリシーを書く

ポリシーは、次のプロトコル（`capx/agent_client/agent_loop.py`）を満たすオブジェクトです。

```python
class Policy(Protocol):
    def reset(self, reset: ResetResult) -> None: ...
    def act(self, obs: ProprioObs | FullObs) -> RawAction: ...
```

`RobotClient`（`capx/agent_client/robot_client.py`）は、`step` の上にサーボやグリッパのプリミティブを用意しています。IK のソルバーは自分で用意して渡します。

```python
from capx.agent_client import RobotClient

robot = RobotClient.start(env, task_id)       # env: EnvClient
robot.move_to_joints(target_joints)            # (7,) の目標関節角へサーボ
robot.close_gripper()
robot.goto_pose(target_pose, ik=my_ik_solver)
```

### 注意点

- **Franka の関節は、観測では8次元、行動では7次元です。** `proprio.joint_pos` は関節7とグリッパ1の `(8,)` ですが、`FrankaRawAction.joints` は `(7,)` です。
- **`mujoco==3.5.0` を使ってください。** 3.10 以上では robosuite が壊れ、`env_crash` になります。
- **LIBERO は専用の venv（`.venv-libero`）が必要です。** 互換性のない robosuite を含むためです。必要なのはバックエンドのプロセスだけです。

---

## CaP-X を単体で使う

コンペティションの評価環境を使わずに、CaP-X のタスクを直接動かすこともできます。

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model "google/gemini-3.1-pro-preview"
```

LLM のプロキシは [docs/configuration.md](docs/configuration.md)、Docker の開発用コンテナは [docker/README.md](docker/README.md) を見てください。

> **コンペでの注意:** コンペで使えるモデルは、[ルールブック §7](PRELIM_RULEBOOK.md#7-使えるモデル) の条件を満たすものだけです。GPT・Claude・Gemini などのクローズドモデルは、評価にもデータ生成にも使えません（§7.1）。この例のモデルは元の CaP-X のものです。

## ドキュメント

| ガイド | 内容 |
| ----- | ---- |
| [予選ルールブック](PRELIM_RULEBOOK.md) | 予選のルール、評価のしくみ、提出物の形式、インターフェースの仕様 |
| [提出の手順](docs/submission.md) | 手元での評価、イメージの作成、HuggingFace への提出 |
| [タスクパック](docs/taskpacks.md) | `pack.yaml` の形式 |
| [評価環境の運用](docs/operations.md) | 評価キューの構成、設定、HuggingFace からの受け付け、別のクラスタへの移行 |
| [コンテナ](docker/README.md) | Docker Compose の構成、開発用コンテナ |
| [ネイティブセットアップ](docs/native-setup.md) | ホストへの uv インストール、知覚サーバー、LLM プロキシ |
| [設定](docs/configuration.md) | YAML の形式、CLI フラグ、LLM プロバイダ |
| [環境の追加](docs/adding-environments.md) | シミュレータ、タスク環境、YAML 設定の作り方 |
| [API の追加](docs/adding-apis.md) | 新しいロボット制御 API の実装と登録 |
| [LIBERO-PRO タスク](docs/libero-tasks.md) | セットアップ、LIBERO タスクの実行、スイート一覧 |
| [BEHAVIOR タスク](docs/behavior-tasks.md) | セットアップ、R1Pro のタスク、環境変数 |
| [開発](docs/development.md) | テスト、lint、既知の問題 |
| [実機 Franka Panda](docs/real-franka.md) | robots_realtime での立ち上げ |
| [RL 学習](docs/rl-training.md) | GRPO/VeRL による CaP-RL |
| [スキルライブラリのコンパイル](scripts/skill_library_compilation/README.md) | 評価結果の分析、スキルライブラリの構築 |

---

## ライセンス

このプロジェクトは [MIT ライセンス](LICENSE) の下で公開されています。
