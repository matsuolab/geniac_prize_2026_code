# コンテナ

このディレクトリには2種類のコンテナがあります。

- **評価用の構成**（`docker compose`）: 提出物・gateway・シミュレータごとのバックエンドを
  別のコンテナに分けたもので、本番の評価と同じ形です。提出物を手元で確かめるのに使います。
- **開発用コンテナ**（`build.sh` と `run.sh`）: CaP-X の `capx/envs/launch.py` を直接動かす、
  1つにまとまったコンテナです。

## 評価用の構成

| ファイル | 役割 |
|---|---|
| `Dockerfile.gateway` | gateway。試行回数・seed・採点を管理し、タスクを `TaskSpec.simulator` に従って各バックエンドへ振り分けます（`CAPX_BACKENDS`）。シミュレータは import しないので CPU だけで動きます。エントリポイントは `serve_environment.py`。 |
| `Dockerfile.env-robosuite`、`Dockerfile.env-libero`、`Dockerfile.env-behavior` | シミュレータごとのバックエンド。エントリポイントは `serve_env_backend.py`（`CAPX_SIMULATOR` で選ぶ）。GPU が必要です。 |
| `Dockerfile.submission` | 提出物の例。`EnvClient` で gateway に接続し、エージェントのループを回します。エントリポイントは `run_submission.py`。 |
| `../docker-compose.yml` | 上の各コンテナを、2つの内部ネットワークでつなぎます。 |

### ネットワーク

外へ出る経路のない（`internal: true`）ネットワークが2つあります。

- `policy_network`: 提出物と gateway だけをつなぎます。提出物はこれにしか接続しないので、
  gateway 以外には届きません（シミュレータにも、インターネットにも）。
- `sim_network`: gateway と各バックエンドをつなぎます。シミュレータの生の観測が流れるので、
  提出物は接続しません。

提出物のコンテナに、外へ出られるネットワークをつないではいけません。

本番の評価キューは Docker ではなく Singularity で動き、提出物のコンテナにはネットワークを
与えず、gateway のソケットだけを渡します。提出物から見た違いは、`EnvClient` の通信が TCP か
ソケットかだけです（[PRELIM_RULEBOOK.md](../PRELIM_RULEBOOK.md) §4.1）。

### 実行する

```bash
# リポジトリのルートで。Robosuite・LIBERO・BEHAVIOR のバックエンドをすべて立てる
CAPX_TASKPACK=taskpacks/original docker compose up --build

# BEHAVIOR のアセット（約37GB）を、名前付きボリューム behavior-assets に一度だけ取得する。
# インターネットに出るので内部ネットワークの外で動く。実行すると BEHAVIOR データセットの
# ライセンスに同意したことになる
docker compose build env-behavior
docker compose --profile assets run --rm fetch-behavior-assets

# 公開パックの BEHAVIOR タスクだけを official モードで動かす
CAPX_SIMULATORS=behavior CAPX_MODE=official CAPX_BACKENDS=behavior=env-behavior:8100 \
    docker compose up --build --no-deps --abort-on-container-exit --exit-code-from submission \
    gateway env-behavior submission

# BEHAVIOR を使わない
CAPX_BACKENDS=robosuite=env-robosuite:8100,libero=env-libero:8100 \
    docker compose up --build gateway submission env-robosuite env-libero
```

提出物は `CAPX_ENV_HOST=gateway`、`CAPX_ENV_PORT=8000` で gateway に接続します。

- `env-behavior` は Blackwell（sm_120）の GPU 向けで、OmniGibson 3.9.2 と Isaac Sim 5.1.0 の
  上に作ります。1つあたり約12GB の GPU メモリを使います。
- `run_submission.py` は、配線を確かめるための単純なポリシーを動かすだけです。参照実装
  （CaP-Agent0）を動かすときは、エントリポイントを
  `python -m capx.submission.run --config-path env_configs/gateway/runner.yaml` にし、
  `Dockerfile.submission` の中でコメントアウトしてある依存（vLLM、SAM3、GraspNet、PyRoKi）を入れます。

### シミュレータを追加する

各シミュレータは互いに合わない依存を持つので、それぞれ専用のイメージにします。

1. そのシミュレータの `LowLevelEnv` のアダプタとファクトリを書く
2. その venv を入れる `Dockerfile.env-<sim>` を書く
3. `docker-compose.yml` に `env-<sim>` を足し、`CAPX_BACKENDS` に加える

gateway は `TaskSpec.simulator` で振り分けるので、変更はいりません。

## 開発用コンテナ

```bash
GPU_ARCH=ada bash docker/build.sh   # blackwell（既定）| ada | hopper | ampere
bash docker/run.sh
```

- `GPU_ARCH` は実際の GPU の世代に合わせてください（`nvidia-smi` で確認）。CUDA 拡張
  （sam3、contact_graspnet、curobo など）がその世代向けにビルドされます。
- `run.sh` は空いている GPU を1枚選び（`GPUS=device=N` で指定もできます）、ホスト側の
  ポートを空いているものに割り当てます（`LLM_PROXY_PORT`、`WEBUI_PORT` で指定もできます）。
  LLM のプロキシ（`capx/serving/openai_server.py` など）は既定でコンテナの中の 127.0.0.1 だけで
  待ち受けます。ホストから使うときは `--host 0.0.0.0` を付けて起動します。
- `docker/api_keys.yaml` と `hf_token` から API キーと HuggingFace のトークンを読みます。
- `CAPX_HF_CACHE_DIR` で HuggingFace のキャッシュの場所を、`CAPX_SHARED_DATASET_DIR` で
  複数のユーザーが共有するデータセットのディレクトリを指定できます。

コンテナの中で、シミュレータごとに一度だけ環境を作ります。リポジトリはマウントされているので、
作った venv はコンテナを作り直しても残ります。

```bash
bash /workspace/docker/setup-robosuite.sh
bash /workspace/docker/setup-libero.sh      # .venv-libero を作る（robosuite と依存が合わないため）
bash /workspace/docker/setup-behavior.sh    # capx/third_party/b1k/.venv を作る（70〜90GB、1〜2時間）

bash /workspace/docker/run-robosuite-smoke.sh
bash /workspace/docker/run-libero-smoke.sh
bash /workspace/docker/run-behavior-smoke.sh
```

API キーなしで試すときは、Qwen3.6-27B をコンテナの中で立てて使えます。

```bash
bash /workspace/docker/run-qwen-server.sh &   # vLLM を :8110 で起動
bash /workspace/docker/run-robosuite-qwen.sh
```

- 初回は重み（約52GB）をダウンロードします。
- SAM3 など gated なモデルを使う設定では、[facebook/sam3](https://huggingface.co/facebook/sam3)
  にアクセスできるトークンが必要です。
- 既定の `GPU_UTIL=0.72`（48GB の GPU を想定）は、Qwen3.6-27B（fp8）を載せられる下限です。
  知覚のサーバーと同じ GPU で足りないときは、別の GPU に分けてください。
