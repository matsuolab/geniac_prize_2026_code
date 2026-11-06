# ネイティブ（Dockerを使わない）セットアップ — アップストリーム CaP-X フレームワーク

これは、アップストリームの CaP-X における開発環境のセットアップ方法です。すべてをホスト上に
直接、`uv` を使ってインストールします。**コンペティションのためだけであれば、通常はここに
書かれている内容は不要です** ── Docker Compose のスタックが、シミュレータごとに専用のコンテナ
を構築してくれます（トップレベルの [README](../README.md) と
[docker/README.md](../docker/README.md) を参照してください）。このガイドが必要になるのは、
コンテナの外でフレームワーク自体をいじりたいとき、対話的な Web UI を動かしたいとき、あるいは
アップストリームの CaP-Bench／CaP-RL のワークフローを再現したいときです。

## インストール

CaP-X は依存関係の管理に [uv](https://docs.astral.sh/uv/) を使います。**Python 3.10** と
**CUDA 対応 GPU** が必要です。

```bash
git clone --recurse-submodules https://github.com/matsuolab/geniac_prize_2026_code && cd geniac_prize_2026_code

# すでに --recurse-submodules なしでクローン済みの場合:
git submodule update --init --recursive

# uv のインストール（未導入の場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.10 && uv venv -p 3.10

# 基本のインストール
uv sync
```

> **補足:** 基本のインストールであっても、すべてのサブモジュールを含めてクローンしておく
> 必要があります。uv のユニバーサル解決には、たとえ導入しない extras の裏にあるパス依存
> 関係であっても、そのビルドメタデータが必要になるためです。基本の `uv sync` に CUDA の
> devel ツールチェーンは不要ですが、インストール時に CUDA 拡張をコンパイルする extras
> （例えば `--extra curobo`）では、ホスト側に `CUDA_HOME`/`nvcc` が必要です。

### シミュレータごとのセットアップ

インストールするシミュレータ系統は**1つだけ**選んでください。Robosuite（1.5.0）と
LIBERO（`robosuite==1.4.0` を要求）は競合します。

#### Robosuite

```bash
uv sync --extra robosuite
```

#### LIBERO-PRO

LIBERO には**専用の仮想環境**が必要です。

```bash
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet
```

130以上ある LIBERO タスクの実行方法は [libero-tasks.md](libero-tasks.md) を参照してください。

#### BEHAVIOR（Isaac Sim）

BEHAVIOR のタスクは、OmniGibson 経由で NVIDIA Isaac Sim 上で動作します。Python 3.10 と
CUDA 12.x が必要です。

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset          # OmniGibson、Isaac Sim、BDDL、cuRobo をインストールし、アセットをダウンロードする
cd ../../..                        # リポジトリのルートに戻る

# インストール後の修正 — cuRobo の JIT ヘッダをコピーする（b1k の venv を有効化した状態で実行）
source capx/third_party/b1k/.venv/bin/activate
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

> `--dataset` フラグは、ロボットのアセット、BEHAVIOR-1K のシーン／物体アセット、2025年
> チャレンジのタスクインスタンスをダウンロードします。NVIDIA Isaac Sim の EULA と
> BEHAVIOR データセットのライセンスへの同意を求められます。自動同意させたい場合は
> `--accept-dataset-tos` を追加してください。

ヘッドレスサーバーの場合:
```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
# 重複した Vulkan ICD が存在する場合は削除する（マルチGPU環境でセグフォルトの原因になる）
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json
```

タスクの詳細と想定されるベースラインは [behavior-tasks.md](behavior-tasks.md) を参照して
ください。

### オプションの extras

```bash
uv sync --extra verl             # VeRL/GRPO による強化学習
uv sync --extra contactgraspnet  # Contact-GraspNet による把持計画
uv sync --extra curobo           # cuRobo（GPUによる高速な IK・モーションプランニング。CUDA が必要）
```

## クイックスタート

### 1. 知覚サーバー（自動起動）

知覚サーバー（SAM3、ContactGraspNet、PyRoKi）は、評価を実行すると YAML の設定に従って
**自動的に起動**します。ほとんどの設定では、手動でのセットアップは不要です。

> **SAM3 の認証について:** SAM3 の重みには HuggingFace へのアクセス権が必要です。
> [SAM3 のリポジトリ](https://github.com/facebookresearch/sam3) でアクセスをリクエストし、
> `huggingface-cli login` でローカルに認証してください。重みは初回ダウンロード後、
> キャッシュされます。

サーバーを事前に起動しておきたい場合（例えば複数の評価実行で共有したい場合）:

```bash
# GPU の自動割り当てで SAM3 + GraspNet + PyRoKi を起動する
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

`--dry-run` を付けると、割り当て内容だけを事前に確認できます。その他のプロファイル:

```bash
--profile full      # すべての知覚サーバー（SAM3、GraspNet、PyRoKi、OWL-ViT、SAM2）
--profile minimal   # PyRoKi のみ（oracle／特権的な評価用）
```

### 2. LLM プロキシのセットアップ

評価ハーネスは、OpenAI 互換の API を公開するローカルプロキシ経由で LLM に問い合わせます。

```bash
# OpenRouter（openrouter.ai/keys でキーを取得）
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

> **補足:** `.openrouterkey` は git の管理対象外です。設定のデフォルトのサーバー URL は
> `http://127.0.0.1:8110/chat/completions` です。

その他のプロバイダのオプション（OpenRouter、vLLM、カスタム）については
[configuration.md](configuration.md) を参照してください。

#### ローカルのオープンウェイトモデル: Qwen3.6-27B（vLLM）

プロキシの代わりに、`Qwen/Qwen3.6-27B` をローカルで動かして、そこにハーネスを向けることも
できます。API キーは不要です。

```bash
# 1. モデルを配信する（fp8、48GB の Ada/Hopper/Blackwell GPU 1枚に収まる。初回実行時に .venv-serve が構築される）
bash docker/run-qwen-server.sh &            # :8110 で OpenAI 互換エンドポイントを公開

# 2. （S3 系／*_reduced_api* 系の設定のみ）:8122 で Molmo のグラウンディングサーバーを起動
bash docker/run-molmo-server.sh &

# 3. ロールアウト／評価の実行（CONFIG, TRIALS, WORKERS, TEMP は環境変数で上書き可能）
bash docker/run-robosuite-qwen.sh
```

はまりやすいポイント（スクリプト側ではすでに対処済みですが、手動セットアップの場合のために
記載します）:
- Qwen3.6 のアーキテクチャ（`qwen3_5`）には **vLLM ≥0.16 と transformers ≥5** が必要です。
  一方このリポジトリの `pyproject.toml` は `override-dependencies=["transformers<5.0"]` を
  固定しているため、配信用の venv は `uv pip install --isolated`（またはリポジトリの外から）
  でインストールしてください。
- 最近の vLLM の wheel は CUDA-13 向けにビルドされています。ホストのドライバは **R580 以上**、
  torch は `cu130` 版（`--torch-backend=cu130`）である必要があります。そうでないと
  `libcudart.so.13` のインポートエラーになります。
- vLLM のエンドポイントは `http://127.0.0.1:8110/v1/chat/completions` です ── プロキシとは
  違い `/v1` が付く点に注意してください。
- 複数ターンの設定には `--max-model-len ≥16384` が必要です（2ターン目のプロンプトにはコード
  とコンソール出力が含まれるためです）。
- 生成が遅い場合: 応答が返り始めるまでの待ち時間は、リクエストごとに `CAPX_LLM_TIMEOUT` 秒です
  （デフォルト 900）。接続できない・タイムアウト・429/502/503/504 のときは間隔をあけて再試行します。
  1回の問い合わせは、再試行と応答の読み込みを含めて、最初のリクエストから
  `CAPX_LLM_RETRY_DEADLINE_S` 秒（デフォルト 1800）で打ち切ります。ほかのエラー
  （400、404、500 など）は再試行せず、すぐに例外になります。

### 3. 評価の実行

```bash
# Robosuite: 単一ターンのベンチマーク（100試行、12並列ワーカー）
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model "google/gemini-3.1-pro-preview"

# Robosuite: 視覚差分検出を使う複数ターン
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml \
    --model "google/gemini-3.1-pro-preview"

# LIBERO-PRO: 空間タスク（.venv-libero が必要）
source .venv-libero/bin/activate
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model "google/gemini-3.1-pro-preview"

# BEHAVIOR: R1Pro によるラジオの拾い上げ（20試行）— b1k の venv が必要
source capx/third_party/b1k/.venv/bin/activate
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
    --model "google/gemini-3.1-pro-preview"

# 対話的な Web UI
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --web-ui True
# 表示された http://127.0.0.1:8200/#token=... を開く（127.0.0.1 でだけ待ち受けるので、
# 別のマシンからは ssh -L 127.0.0.1:8200:127.0.0.1:8200 <host> でポートを転送する）

# 回帰テスト
./scripts/regression_test.sh quick    # 10試行のスモークテスト（約30秒）
./scripts/regression_test.sh test1    # 単一ターンのフルテスト（約3分）
```

> **ヒント（BEHAVIOR）:** Isaac Sim は GPU の選択に `CUDA_VISIBLE_DEVICES` ではなく
> `OMNIGIBSON_GPU_ID` を使います。マルチGPU環境で最良のパフォーマンスを得るには、知覚
> サーバーを別の GPU で動かしてください（例えば評価には `OMNIGIBSON_GPU_ID=0` を使い、
> SAM3/GraspNet は `CUDA_VISIBLE_DEVICES=1` で事前に起動しておく）。ヘッドレス評価では
> `OMNI_KIT_ACCEPT_EULA=YES` と `OMNIGIBSON_HEADLESS=1` を設定してください。
