# CaP-RL: Code-as-Policy エージェントのための強化学習

原文（英語）: [rl-training.md](https://github.com/capgym/cap-x/blob/main/docs/rl-training.md)

CaP-RL は、検証可能な環境報酬を用いたオンポリシー強化学習を可能にし、より優れたロボット操作プログラムを生成できるようコーディングエージェントを学習させます。CaP-Gym タスクで言語モデルを事後学習するために、Group Relative Policy Optimization（GRPO）を使用します。

> **主な結果:** RL による事後学習を行ったエージェントは、実機の Franka Panda ロボットにゼロショットで展開した場合でも、**キューブ持ち上げで 84%**、**キューブ積み上げで 76%** の成功率を維持しました。追加のファインチューニングは不要です。

オンポリシー (On-policy) : 現在の方策 (policy) を用いて集めたデータを、方策の更新（学習）に利用すること  
GRPO: 代表的な強化学習アルゴリズムの一つ  
ゼロショット (zero-shot) : ここでは追加のファインチューニングを行わないこと


---

## 目次

- [セットアップ](#セットアップ)
- [学習](#学習)
  - [学習スクリプトの処理内容](#学習スクリプトの処理内容)
  - [設定](#設定)
- [評価](#評価)
  - [チェックポイントのマージ](#1-fsdp-チェックポイントのマージ)
  - [vLLM によるサービング](#2-vllm-によるサービング)
  - [評価の実行](#3-評価の実行)
- [利用可能なタスク](#利用可能なタスク)

---

## セットアップ

flash-attn の wheel には Python 3.12 が必要なため、CaP-RL にはメインプロジェクトとは**別の仮想環境**が必要です。

flash attention : attention の GPU メモリ消費を抑え、高速化する代表的なアルゴリズム

```bash
# 専用の RL 環境を作成して有効化（flash-attn には Python 3.12 が必要）
uv venv .venv-rl --python 3.12
source .venv-rl/bin/activate

# RL とシミュレーションの依存関係をインストール（--active は有効化済みの venv を対象とする）
uv sync --active --extra verl --extra robosuite

# Weights & Biases にログイン（学習メトリクスのダッシュボード）
wandb login

# Hugging Face にログイン（モデルチェックポイントのダウンロード）
huggingface-cli login
```

**要件:**

- Python 3.12（flash-attn の wheel には cp312 が必要）
- CUDA 対応 GPU（H100、A100、RTX 4090 で動作確認済み）
- 実験追跡用の [Weights & Biases](https://wandb.ai/) アカウント
- モデルへのアクセス用の [Hugging Face](https://huggingface.co/) アカウント


wandb : wandb は、モデルの学習の際に、 loss の遷移などの可視化をはじめとして、有用な補助ツールとして使われる。

---

## 学習

各タスクは GRPO を使用して個別に学習します。`MODEL_PATH` に Hugging Face モデル、`DATA_ROOT` に出力先、`DATA_SOURCE` に環境名を設定します。

（各タスク共通）
`CUDA_VISIBLE_DEVICES=5 N_GPUS=1` のように、コマンドの中に使える GPU と数を明示しておくと良いと思います。（必須ではない）
0.5B では動作確認済み (2026/07/20)

### Cube Lift (キューブの持ち上げ)

```bash
source .venv-rl/bin/activate
MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct \
DATA_ROOT=output/franka_cube_lift \
DATA_SOURCE=franka_lift_code_env \
bash scripts/train_franka_grpo.sh
```

### Cube Stack (キューブの積み上げ)

```bash
source .venv-rl/bin/activate
MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct \
DATA_ROOT=output/franka_cube_stack \
DATA_SOURCE=franka_robosuite_pick_place_code_env \
bash scripts/train_franka_grpo.sh
```

###.  Spill Wip(こぼれた液体の拭き取り)

```bash
source .venv-rl/bin/activate
MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct \
DATA_ROOT=output/franka_robosuite_spill_wipe_code_env \
DATA_SOURCE=franka_robosuite_spill_wipe_code_env \
bash scripts/train_franka_grpo.sh
```

学習ログとチェックポイントは `DATA_ROOT` に保存されます。W&B ダッシュボードで学習の進捗を確認できます。

### 設定

異なるタスクやモデルで学習する場合は、以下の主要な環境変数を上書きします。

| 変数 | デフォルト | 説明 |
| --- | --- | --- |
| `MODEL_PATH` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Hugging Face のモデル ID またはローカルパス |
| `DATA_ROOT` | `$HOME/data/capx/franka` | データセットとチェックポイントの出力ディレクトリ |
| `DATA_SOURCE` | `franka_pick_place_code_env` | 環境名（[利用可能なタスク](#利用可能なタスク)を参照） |
| `N_GPUS` | 自動検出 | 使用する GPU の数 |
| `GROUP_SIZE` | `15` | GRPO のグループサイズ（プロンプトごとのロールアウト数） |
| `TRAIN_TEMPERATURE` | `1.0` | 学習中のサンプリング温度 |
| `TRAIN_DATASET_SIZE` | `256` | 学習プロンプト数 |
| `VAL_DATASET_SIZE` | `256` | 検証プロンプト数 |
| `PYROKI_PORT` | `8116` | PyRoKi IK サーバーのポート |

---

## 評価

評価には、分散チェックポイントのマージ、モデルのサービング、評価ハーネスの実行という 3 つの手順が必要です。

### 1. FSDP チェックポイントのマージ

学習後、GRPO は FSDP でシャーディングされたチェックポイントを保存します。これらを標準の Hugging Face 形式にマージします。

```bash
# チェックポイントは DATA_ROOT/checkpoints/capx/<experiment_name>/ 以下に保存される
export CKPT_PATH=<path-to-global_step_N/actor>
export TARGET_DIR=${CKPT_PATH%/actor}/hf_checkpoint

source .venv-rl/bin/activate
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CKPT_PATH \
    --target_dir $TARGET_DIR
```

### 2. vLLM によるサービング

マージしたチェックポイントを OpenAI 互換 API サーバーとして起動します。

```bash
source .venv-rl/bin/activate
python -m capx.serving.vllm_server --model $TARGET_DIR
```

デフォルトでは、サーバーは `http://localhost:8000` で起動します。

### 3. 評価の実行

CaP-X の評価ハーネスを、起動したモデルに接続します。`--model` フラグは、手順 2 で vLLM に渡したパスと一致している必要があります。

注意：評価コマンドは 2 を実行したターミナルと別のターミナルで実行することになる。その際、2 で設定した $TARGET_DIR は引き継がれないので、
```
export CKPT_PATH=<path-to-global_step_N/actor>
export TARGET_DIR=${CKPT_PATH%/actor}/hf_checkpoint
```
を実行しておく。これをしていない ($TARGET_DIR が設定されていない)と、以下のコマンドで Unrecognized options: 100 のようなエラーが出る。

```bash
source .venv-rl/bin/activate
python capx/envs/launch.py \
    --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml \
    --server-url http://127.0.0.1:8000/chat/completions \
    --model $TARGET_DIR \
    --total-trials 100 \
    --server-url http://127.0.0.1:8000/v1/chat/completions
```

---

## 利用可能なタスク

| タスク | `DATA_SOURCE` | 設定 | 説明 |
| --- | --- | --- | --- |
| Cube Lift | `franka_lift_code_env` | `cube_lifting/` | 赤いキューブを高さのしきい値より上まで持ち上げる |
| Cube Stack | `franka_robosuite_pick_place_code_env` | `cube_stack/` | 赤いキューブを緑のキューブの上に積む |
| Spill Wipe | `franka_robosuite_spill_wipe_code_env` | `spill_wipe/` | スポンジでこぼれた液体を拭き取る |

その他のタスクと YAML 設定の一覧については、`env_configs/` を参照してください。
