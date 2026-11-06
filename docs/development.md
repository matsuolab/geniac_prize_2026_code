# 開発ガイド

[原文（英語）: development.md](https://github.com/capgym/cap-x/blob/main/docs/development.md)

CaP-Xに変更を加えたときに、動作確認・コード品質の確認・シミュレータの準備を行うためのガイドです。

## テスト

### ユニットテスト

```bash
uv run pytest tests/test_environments.py -q
```

特定のテストだけを実行する場合:

```bash
uv run pytest tests/test_environments.py::test_franka_pick_place_code_env -q
uv run pytest tests/test_environments.py::test_franka_nut_assembly_code_env -q
```

`tests/test_environments.py`のテストは実際にシミュレータを起動します。必要なシミュレータ（RobosuiteやBEHAVIOR）がインストールされていない環境では、そのテストはスキップされます。

### オラクルコードのテスト

特定の環境で、あらかじめ用意された正解動作（オラクルコード）がどの程度の性能を出すかを確認します。

```bash
uv run tests/test_environments.py --env-name YOUR_ENV_NAME
```

`--video-path out.mp4`を付けると、実行の様子を動画として保存します。

### 回帰テスト（期待報酬）

マージする前に次の環境を実行し、得られた報酬の平均が期待値と一致することを確認してください。報酬は、タスクがどの程度成功したかを表す評価値です。

```bash
uv run capx/envs/launch.py --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml
# 期待される平均報酬: 約0.99

uv run capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml
# 期待される平均報酬: 約0.90

uv run capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml
# 期待される平均報酬: 約0.50

uv run capx/envs/launch.py --config-path env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml
# 期待される平均報酬: 約0.15

uv run capx/envs/launch.py --config-path env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml
# 期待される平均報酬: 約0.25

uv run capx/envs/launch.py --config-path env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml
# 期待される平均報酬: 約0.20
```

## Lint

```bash
ruff check          # Lint
ruff check --fix    # 自動修正
ruff format         # フォーマット
```

コントリビューションを行うときは、Lintにruff（自動インストール済み）を使用してください。詳しくは[ruffのドキュメント](https://docs.astral.sh/ruff/tutorial/#getting-started)を参照してください。

## SAM3へのアクセス

facebookresearchによるSAM3は、サードパーティ連携を通じてパッケージとしてインストールします。SAM 3を使う前に、[SAM 3のHugging Faceリポジトリ](https://huggingface.co/facebook/sam3)でチェックポイントへのアクセスを申請してください。

アクセスが許可されたら、アクセストークンを作成し、チェックポイントをダウンロードできるよう認証する必要があります（例: `hf auth login`）。

## LIBERO-PROのインストール

```bash
uv sync --extra libero --extra contactgraspnet
```

ヘッドレスサーバーでは、EGLレンダリングもインストールしてください。

```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
```

タスクの詳しいリファレンスは、[libero-tasks.md](libero-tasks.md)を参照してください。

## BEHAVIORのインストール（Isaac Sim）

現在のDocker環境では、コンテナ内のリポジトリルートからセットアップスクリプトを実行してください。

```bash
bash /workspace/docker/setup-behavior.sh
```

このスクリプトはPython 3.10のBEHAVIOR用venvを作成し、次のものをまとめてセットアップします。

- BDDL、OmniGibson、Isaac Sim 4.5.0
- cuRobo、PyRoKi
- SAM3とContactGraspNetの実行依存関係
- ロボットアセット、BEHAVIOR-1Kアセット、2025年チャレンジのタスクインスタンス

また、Isaac Simのwebsockets競合の修正、cuRoboのJITヘッダー、CaP-Xの実行時依存関係もこのスクリプトが準備します。

共有データセットがすでに用意されている場合は、ダウンロードをスキップできます。

```bash
SKIP_DATASET=1 bash /workspace/docker/setup-behavior.sh
```

> **注意:** 初回実行時、cuRoboはCUDAカーネルをJITコンパイルします（3〜5分）。Isaac Simも初回にシェーダーをコンパイルするため、起動までさらに約3分かかります。

### 前提条件

- Python 3.10（Isaac Simのwheelはcp310専用です）
- CUDA 12.xを利用できるNVIDIA GPU（ドライバ550以上）
- ヘッドレスレンダリング用の`libegl1`と`libgl1`（上記参照）

### 環境変数

画面のないヘッドレスサーバーでは、実行前に次を設定してください。

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
```

## リポジトリ内で管理しているサブモジュール

再現性のあるオフラインテストを行うため、一部の上流リポジトリをこのリポジトリ内で管理しています。クローン後にサブモジュールを初期化してください。

```bash
git submodule update --init --recursive
```

## 注意点・既知の問題

1. MuJoCoでは、衝突が重要なボディに`condim="4"`を使用してください。既定値は3で、物体が滑る原因になることがあります。[MuJoCoのドキュメント](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-condim)を参照してください。
2. コード実行のサンドボックスはローカルで、ある程度の安全性を備えています。より強い分離が必要な場合は、Dockerまたはnsjailを使用してください。
