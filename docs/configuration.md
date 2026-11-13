# 設定リファレンス

原文（英語）: [configuration.md](https://github.com/capgym/cap-x/blob/main/docs/configuration.md)

## CLI フラグ

コマンドラインから任意の YAML 設定を上書きできます。

なお、このコマンドを使用せずに vim や VSCode などを用いて書き換えても良いです。

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path <config.yaml> \
    --model google/gemini-3.1-pro-preview \
    --server-url http+unix://model.sock/chat/completions \
    --temperature 1.0 \
    --total-trials 100 \
    --num-workers 12 \
    --record-video True
```

| フラグ | デフォルト | 説明 |
| --- | --- | --- |
| `--config-path` | *（必須）* | YAML task config へのパス |
| `--model` | `google/gemini-3.1-pro-preview` | モデル名 |
| `--server-url` | `http+unix://model.sock/chat/completions` | LLM エンドポイント。ローカルのサーバーは `http+unix://<ソケット>/<パス>`、外部の API は `https://`（[README](../README.md#注意点)） |
| `--temperature` | `1.0` | サンプリング温度 |
| `--total-trials` | YAML の値 | 評価試行回数 |
| `--num-workers` | YAML の値 | 並列ワーカー数 |
| `--web-ui` | `False` | 対話型 Web UI を起動するかどうか。127.0.0.1 で待ち受け、アクセス用トークン付きの URL を表示する（待ち受けるアドレスは環境変数 `CAPX_WEB_UI_HOST` で変えられる） |
| `--use-oracle-code` | `False` | 人手で作成された参照解を実行するかどうか |

## YAML 設定形式

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

`_target_` キーにより、`capx.envs.configs.instantiate()` を介した Hydra 形式の遅延インスタンス化が可能になります。

### Perception servers（api_servers）

YAML 設定に `api_servers` セクションを含めると、評価開始時に Perception server を**自動起動**できます。

```yaml
api_servers:
  - _target_: capx.serving.launch_sam3_server.main
    device: cuda
    socket: sam3.sock

  - _target_: capx.serving.launch_contact_graspnet_server.main
    socket: graspnet.sock

  - _target_: capx.serving.launch_pyroki_server.main
    socket: pyroki.sock
    robot: panda_description
    target_link: panda_hand
```

各サーバーは `socket` の Unix ソケットだけで待ち受けます。ソケット名だけなら `CAPX_SOCKET_DIR`
（既定は `~/.capx/sockets`、0700）の中に作られます。`port` と `host` は使えません。

ランチャーは以下を自動的に行います。

- ソケットですでに待ち受けているサーバー（外部で起動済みの場合など）をスキップ
- 試行を実行する前に、すべてのサーバーが準備完了になるまで待機
- 終了時にすべてのサーバーを停止

サーバーを個別に管理したい場合（複数の評価実行間で共有する場合など）は、`launch_servers.py` を使用してください。

```bash
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

| profile | サーバー | GPU の要否 |
| --- | --- | --- |
| `default` | SAM3（`sam3.sock`）+ ContactGraspNet（`graspnet.sock`）+ PyRoKi（`pyroki.sock`） | 必要（VRAM 約 5 GB） |
| `full` | default + OWL-ViT（`owlvit.sock`）+ SAM2（`sam2.sock`） | 必要（VRAM 約 14 GB） |
| `minimal` | PyRoKi（`pyroki.sock`）のみ | 不要（CPU のみ） |

## 新しい LLM プロバイダーの追加

CaP-X は、OpenAI 互換の `/chat/completions` エンドポイントを公開するローカルプロキシサーバーを介して言語モデルにクエリを送信します。

### OpenRouter

OpenRouter では、1 つの API キーで Gemini、GPT、Claude、DeepSeek、Qwen などのモデルを利用できます。
なお、 OpenRouter では無料モデルを使用できますが、単位時間あたりに送れるリクエストが限られています。たくさんのリクエストを送る必要のある Cap-X での評価には向いていません。

> **コンペでの注意:** コンペで使えるモデルは、[ルールブック §7](../PRELIM_RULEBOOK.md#7-使えるモデル) の条件を満たすものだけです。GPT・Claude・Gemini などのクローズドモデルは、評価にもデータ生成にも使えません（§7.1）。この例のモデルは元の CaP-X のものです。

1. [openrouter.ai/keys](https://openrouter.ai/keys) で API キーを取得します。
2. プロジェクトルートのファイルに保存します。

   ```bash
   echo "sk-or-v1-your-key-here" > .openrouterkey
   ```

3. プロキシを起動します（複数のキーにわたる自動キーローテーションに対応しています）。

   ```bash
   uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey
   ```

   プロキシは `model.sock` で待ち受けます。`--server-url` の既定値 `http+unix://model.sock/chat/completions`
   でそのまま使えます。

   ```bash
   # 別の名前で立てるとき
   uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --socket other.sock
   ```



### オプション B: vLLM（ローカルモデル）

```bash
uv run python -m capx.serving.vllm_server --model Qwen/Qwen2.5-Coder-7B-Instruct --tensor-parallel-size 4
# model.sock で待ち受ける。--server-url http+unix://model.sock/v1/chat/completions で使う
```

### オプション C: カスタムプロバイダー

プロバイダーは `capx/serving/providers/` 以下に配置され、シンプルな `generate_code` メソッドを実装します。新しいプロバイダークラスを追加することで、Gemini、Claude、Bedrock に拡張できます。

> **注意:** `.openrouterkey` は Git の追跡対象外です。API キーをリポジトリにコミットしないでください。
