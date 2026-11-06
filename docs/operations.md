# 評価キューの運用

主催者向けに、評価キューを立ち上げて提出物を評価し、運用するための手順をまとめます。
提出物が満たすことと評価のルールは [PRELIM_RULEBOOK.md](../PRELIM_RULEBOOK.md)、チームが提出する手順は
[submission.md](submission.md) にあります。ここではそれを繰り返しません。

## 1. 仕組み

評価はすべて評価キューを通ります。キューは共有ディレクトリ（設定の `root`）そのもので、
3種類のプロセスはこのディレクトリを介してだけやり取りします。

```
ログインノード（GPU 不要）                         計算ノード（GPU 1枚 / worker）
capx evalq watch      HF を確認 → 検証 → 取得 ─┐
capx evalq submit     手元の .sif を直接入れる ─┼─▶ <root>/queue ─▶ capx evalq worker
capx evalq supervise  worker ジョブを維持       │                    gateway ─ env-<sim> ─ 提出コンテナ
capx evalq publish    リーダーボードを HF へ ◀──┘ <root>/results
```

| `<root>` の下 | 中身 |
|---|---|
| `incoming/<team_id>/<key>/` | watcher が HF から取得したイメージと `submission.yaml`。`<key>` は `submission.yaml` の中身の sha256 の先頭12桁。結果が出ると worker が消す（§5） |
| `queue/{pending,claimed,done,failed}/` | ジョブ1件に JSON 1つ。置かれているディレクトリが状態 |
| `runs/<run_id>/<subset>/` | 1回の評価のログと作業ファイル。各チームの最新の評価と、運営側の失敗で終わった評価の分を残す（下記） |
| `results/<run_id>.json` | 結果。却下も `status: rejected` でここに残る |
| `state/seen/` | watcher が処理済みの提出の記録 |
| `state/deferred/<team_id>.json` | 評価中の提出の後ろで待っている、そのチームの最新の提出 |
| `state/retrying/` | HF の一時的な障害でやり直している提出と、最初に失敗した時刻（§5） |
| `state/ranking/` | 順位評価の提出の検査結果（§7）。`heads/` に、締切の前に最後に見た各チームの `main` と、締切の時点の `main` |
| `weights/<owner>--<name>/<commit>/` | 提出物が `weights` で申告したモデル |

`runs/<run_id>/` は、同じチームの次の評価が始まるときに worker が消します。消すのは結果が出ている評価だけで、
`results/` と `queue/` の記録は残ります。実行中の評価のほか、次のものも消さないので、運営が見てから手で消します。

- 運営側の失敗（`internal_error`、`no_score`）で終わった評価。イメージと同じく、原因を調べて入れ直せるように残します。
- worker が止められて結果を出さなかった評価。実行中でないことを確かめてから消します。

順位評価では何も消えません。各チーム1回で、`--again` で入れ直すのは運営側の失敗のときだけだからです。

キューは提出順に処理します。1チームが同時に持てる提出（待機中＋評価中）は
`watcher.max_active_per_team`（既定1、0で無制限）件です。上限を超える push は、そのチームの一番古い
**待機中**の提出を取り消して（`status: cancelled`）入れ替えます。取り消すのは新しい提出が検査を通り、
イメージの転送が終わってからです。**評価中**の提出は止めず、新しい方は `state/deferred/` に記録して
評価が終わった次の確認で入れます。`evalq submit` で手で入れたものはこの上限に数えず、取り消しもしません。worker は `slurm.workers` 本まで立ち、それぞれ GPU を1枚使います。
別ホストで動くシミュレータ（`remote`、§6）を使う subset があるときは1本だけです。

### 隔離

worker は1提出・1タスクパック（subset）ごとに、gateway、シミュレータのバックエンド、
提出コンテナを同じ計算ノードで起動します。

- 提出コンテナは `singularity run|exec --net --network=none --containall --nv` で動きます。
  ネットワークは loopback だけで、ホストのファイルシステムは明示した bind 以外見えません。
- bind するのは gateway のソケットのディレクトリ（`/capx/run`）と申告した重み（どちらも読み取り専用）、
  書き込み用の `/capx/out` と `/capx/home` だけです。ほかに書き込めるのはコンテナの `/tmp` と `/var/tmp`
  （Singularity のメモリ上の領域。合わせて 64 MiB）です。
- ソケットは `gw/`（gateway）と `be/`（バックエンド）の2つのディレクトリに分かれていて、
  提出コンテナに mount するのは `gw/` だけです。提出物は生の観測や報酬に届きません。
- 主催者側のコンテナにはそれぞれ専用の `/tmp`（`runs/.../tmp-<name>`）を bind します。
- 環境変数は `--env` ではなく env ファイル（`runs/.../<name>.env`）で渡します。
  提出物の `env` より主催者側の変数（`CAPX_ENV_SOCKET`、`HF_HUB_OFFLINE=1` など）が優先します。
  Singularity はこのファイルをシェルとして読むので、値はすべてクォートして書きます。
- worker が起動するコマンド（`singularity`、`ssh`、`rsync`）には、worker の環境変数のうち
  `PATH`、`HOME`、`LANG` だけを渡します（`runner.HOST_ENV`）。Singularity は env ファイルの変数を
  自分の環境で展開し、`SINGULARITY_BIND` などの設定も環境から読むためです。
- 提出コンテナの出力（`submission.log`）は先頭と末尾の 32 MiB ずつだけを残します。
- `/capx/out` と `/capx/home` は共有ディレクトリの `runs/` の下にあるので、書ける量に上限があります。
  runner が約 30 秒ごとに2つを合わせて数え、`limits.submission_storage_gb`（既定 20 GiB）か
  `limits.submission_storage_files`（既定 100,000 個。ファイルとディレクトリ）を超えたら提出コンテナを止めます。
  提出物のプロセスが終了したときと同じく、それまでに記録された分で採点し、`failed`（`submission_storage_exceeded`）に
  なります。読めないディレクトリや 256 階層より深いディレクトリ、開いているファイルを確かめられないプロセスが
  あって数えられないときも同じです。
  - 数えるのは割り当てられたブロックで（`/capx/out` と `/capx/home` そのもののブロックも含みます）、ハードリンクは
    1回です。シンボリックリンクはたどりません。ディレクトリは名前で1つずつ開くので、数えている途中で
    シンボリックリンクに差し替えられても外へは出ません。
  - 消したあとも提出物のプロセスが開いたままのファイルも数えます。見るのはホストの `/proc` ではなく、提出コンテナの
    `/proc` です（コンテナのプロセスの1つから `/proc/<pid>/root/proc` で入ります）。ホストの `/proc` は `hidepid` で、
    dumpable でないプロセス（Singularity の init や、提出物が自分でそうしたもの）を隠すためです。コンテナの
    `/proc` には PID 名前空間のすべてのプロセスが出ます。Singularity の init（PID 1）と他のユーザーのプロセスは
    除き、記述子を読めないプロセスがあれば数えられないものとして扱います。
  - コンテナの `/proc` は、提出物の起動から約1秒ごとに探し、見つけたら開いたままにします。あとでプロセスが
    dumpable でなくなってホストから見えなくなっても、開いた `/proc` からは見えます。起動から60秒たっても見つからない
    とき（最初から dumpable でないときなど）と、開いた `/proc` が読めなくなったときは、数えられないものとして扱います。
  - 消したファイルをメモリにマップしたまま記述子を閉じたものは、root でないと大きさを確かめられないので数えません。
    ルールブック §5 で違反としています。
  - 超えたと分かった時点で、gateway に SIGTERM を送ってジョブを終わらせ（gateway は SIGTERM でジョブを終えてから
    最終の結果を書きます）、提出コンテナは猶予なしで止めます。そのあとの成功は数えません。
  - 提出物の `finish()` が先だったかどうかは、gateway の最終の結果にある `finished_at`（gateway が `finish()` を
    受け付けた時刻）で決めます。`job-finished` の書き込みが遅れても、`finish()` が先なら上限超えにはしません。

この隔離はコンテナランタイムの性質なので、ラウンドの前と、Singularity やドライバを更新した
あとに実機で確かめます。

```bash
sbatch -p <partition> --gres=gpu:1 scripts/evalq/verify-isolation.sbatch
```

### バックエンドと時間

- バックエンドは必要になったときに起動します。gateway が `be/<sim>-<lane>.sock.want` を置くと
  runner がそのレーンのバックエンドを起動し、落ちたバックエンドは次に要求されたときに
  起動し直します。同時に動くのは1シミュレータあたり最大 `max_lanes` 個です。
- ラウンドのルール（`trial_time_s`、`n_trials`、`trial_steplimit`）は各タスクパックの
  `pack.yaml` の `limits` にあり、キューの設定には書きません。
- ジョブ全体の時間は `trial_time_s` × その subset が選んだタスクの試行の総数です。
- ジョブの時計は、提出物が初めて gateway に接続した時点で始まります。期限が来ると runner が
  提出コンテナを止めます。それまでに採点された試行の結果は残ります。
- 接続までの待ち時間は `limits.submission_startup_s`（既定 6 時間）、バックエンドと gateway の
  ソケットが現れるまでの待ち時間は `limits.backend_startup_s`（既定 600 秒）です。
  時間内に接続しなかった提出は採点せず、`failed`（`submission_startup_timeout`）になります。
  時間内にソケットを作らなかったバックエンドは止め、gateway が次に要求したときに起動し直します。

## 2. 準備

### 前提

- Slurm と Singularity（SingularityPRO 4.1 で確認）が使え、ログインノードと計算ノードの両方から
  見える共有ディレクトリがあること。重みとイメージで数百 GB を使います。
- ビルドと評価は必ず `sbatch` で計算ノードに投げます。ログインノードで動かすのはキューの操作、
  watcher、supervise、publish だけです。
- `sbatch` は `singularity` コマンドが無いときに `module load singularitypro/4.1` を実行します。
  モジュール名が違う場合は `export CAPX_SINGULARITY_MODULE=<モジュール名>` をしておきます。

### キュー操作用の venv

`capx evalq` は軽い Python 環境で動きます。重い依存はすべてイメージの中にあります。
Python 3.11〜3.12 を使います（`datetime.UTC` を使うため、3.10 では動きません）。

```bash
python3 -m venv ~/capx-venv
~/capx-venv/bin/pip install --upgrade pip
~/capx-venv/bin/pip install pydantic pyyaml tyro msgpack \
    "huggingface_hub>=1.2" pytest numpy pillow
export CAPX_EVALQ_PYTHON=$HOME/capx-venv/bin/python

cd ~/cap-x-geniac
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq status --help
```

使い方が出れば動いています。以下のコマンドはすべてリポジトリのルートで実行します。

### イメージ

| イメージ | 使う場面 | 必要な submodule |
|---|---|---|
| `gateway` | 常に | なし |
| `env-robosuite` | Robosuite のタスク | `capx/third_party/robosuite` |
| `env-libero` | LIBERO のタスク | `capx/third_party/LIBERO-PRO`、`capx/third_party/libero_dependencies/robosuite` |
| `submission-agent0` | 参照実装（CaP-Agent0）での確認 | `capx/third_party/sam3`、`capx/third_party/contact_graspnet_pytorch` |

```bash
git submodule update --init --recursive <上の submodule>
mkdir -p logs
sbatch -p <partition> singularity/build.sbatch gateway env-robosuite env-libero submission-agent0
```

- 必要なイメージは**1つのジョブにまとめて**渡します。ジョブを分けると、同じ出力先を使うため
  互いの `.sif` を上書きします。
- 出力は `singularity/images/capx-<name>.sif`、ログは `capx-sif-build.<JOBID>.log` です。
  置き場所を変えるときは `sbatch` の前に `export CAPX_IMAGE_DIR=...` します。
- `capx/`、`docker/`、`env_configs/`、`taskpacks/` に git の管理外のファイルがあると、
  イメージに入らないようビルドが止まります。
- GraspNet のチェックポイント（`capx/third_party/contact_graspnet_pytorch/checkpoints/`）の
  `model.pt` が 100 バイト程度なら git-lfs のポインタのままです。そのディレクトリで
  `git lfs pull` してからビルドします。空のままでもビルドは通り、実行時に把持計画だけが動きません。

`capx/` はイメージに焼き込まれるので、変更はイメージを作り直すまで評価に反映されません。
例外は `capx/evalqueue/` で、worker と watcher はホストのチェックアウトから直接読みます。
イメージを要するテスト（`tests/submission` など）は、作業ツリーを参照実装のイメージに bind して
実行します。

```bash
sbatch -p <partition> --gres=gpu:1 scripts/evalq/test-in-image.sbatch tests/submission
```

### 重み

提出コンテナにはネットワークが無いので、モデルは `submission.yaml` の `weights` で申告して
もらいます（形式は [PRELIM_RULEBOOK.md](../PRELIM_RULEBOOK.md) §6.2）。キューは受け付けるとき（`watch` または
`submit` の時点）に `weights/<owner>--<name>/<commit>/` へ取得し、実行時に申告された `path` へ
読み取り専用で mount します。repo と commit が同じなら複数の提出で共有します。

- gated なリポジトリは主催者アカウントのトークン（`CAPX_HF_TOKEN`）で取得します。
  主催者アカウントにアクセス権が無ければ却下になるので、`declares gated weights` という
  WARNING が watcher のログに出たら、そのリポジトリのアクセス権を確認します。
- チームの org の外のリポジトリは、公開されているものだけを受け付けます。主催者のトークンは
  登録前のチームの org なども読めるので、非公開のものを取得させないためです。
- 1提出の重みの合計の上限は `watcher.max_weights_gb`（既定 300、単位は GiB）です。
- 待機中・実行中のどのジョブも使っていない重みは、worker がジョブを1つ終えるたびに消します。
  取得から10分以内のものは残します。

参照実装の manifest は `singularity/capx-submission-agent0.yaml` です。Qwen3.6-27B（約55GB）、
SAM3（約7GB、gated）、両腕のタスクで使う OWLv2 と SAM2.1 を使うので、初回の `submit` は取得に
時間がかかります。

## 3. キューの設定

既定値は `capx/evalqueue/config.py` の pydantic モデルにだけあり、`--config` に渡す YAML には
既定と違う項目だけを書きます。主催者が本番で使う設定ファイル（チームの登録、
共有ディレクトリ）は公開リポジトリの外に置きます。運営は、非公開のタスクパックと設定を
`private/` submodule（運営の非公開リポジトリ）から `git submodule update --init --checkout private` で取得します。パスは、絶対パスと `~` で始まるものは
そのまま、相対パスはリポジトリのルートを基準に解決します。

| 項目 | 既定 | 内容 |
|---|---|---|
| `root` | `~/shared-storage/submission` | 共有ディレクトリ。計算ノードから見える場所 |
| `socket_dir` | `/tmp` | ソケットの親ディレクトリ。ノードローカルで短いパス（AF_UNIX は 107 バイトまで） |
| `mode` / `master_seed` | `official` / 無し（`official` では必須） | gateway のモードと seed の元（下記） |
| `deadline` | 無し | ラウンドの締切（タイムゾーン付き）。この時刻からは何もキューに入れず、待機中の提出は watcher と worker が取り消し、評価中の提出は runner が止めて cancelled にする（結果は出さない） |
| `limits` | 21600 / 600 秒、20 GiB / 100,000 個 | `submission_startup_s`、`backend_startup_s`、`submission_storage_gb`、`submission_storage_files`（§1） |
| `watcher` | | `poll_interval_s`（worker の確認間隔）、`max_active_per_team`（1）、`max_weights_gb` |
| `hub` | | `token_env`（`CAPX_HF_TOKEN`）、`poll_interval_s`（60） |
| `teams` | 登録無し | `repo`、`ref`（`main`）、`path`（`teams/prelim`）。チームの登録を読む場所（§5） |
| `subsets` | `original`（Robosuite と LIBERO） | `name`、`taskpack`、`simulators`（空ならパックの全タスク） |
| `gateway_image` | `singularity/images/capx-gateway.sif` | |
| `simulators` | robosuite・libero（各 `max_lanes: 32`） | `<id>: {image または remote, max_lanes}` |
| `slurm` | | `partition`、`gres`（`gpu:1`）、`cpus_per_task`（32）、`mem`（512G）、`time_limit`、`job_name`、`workers`（2。`remote` のシミュレータを使う subset があるときは 1 だけ。§6） |
| `leaderboard` | 公開しない | `repo`、`token_env`（`CAPX_HF_LEADERBOARD_TOKEN`）、`interval_s`（60、これ未満は不可）、`heartbeat_s`（1800）など（§5.1） |

`master_seed` は各エピソードの seed の元です。seed の求め方はコードと一緒に公開されているので、
この値が分かると提出物が seed を計算できます。ラウンドごとに
`python3 -c 'import secrets; print(secrets.randbits(63))'` で新しく作り、公開リポジトリの外の設定ファイルにだけ
書きます。順位評価の設定には、中間評価とは別の値を使います。

`slurm.partition` が空のときは `part-<主グループ名>` を使います。命名が違うクラスタでは必ず書きます。
`simulators` を書くと既定の辞書ごと置き換わるので、使うシミュレータはすべて書きます。

`max_lanes` は1提出が同時に走らせてよいエピソード数の上限です。実際に何レーン使うかは提出物が
決めます（参照実装は 16）。上限はラウンドの間は変えず、ルールと一緒に公開します。
`mem: 512G` は 16 レーンで長いパックを流せる大きさです。

最小の例（Robosuite だけを1タスクで確かめる設定）です。

```yaml
# ~/capx-queue/queue.yaml
root: ~/capx-queue/root
master_seed: 1234   # 確認用。本番では上のとおり作った値
subsets:
  - name: smoke
    taskpack: ~/capx-queue/pack-smoke
    simulators: [robosuite]
slurm:
  partition: <partition>
  workers: 1
```

`~/capx-queue/pack-smoke/pack.yaml` は `taskpacks/original/pack.yaml` を元に、`pack_id` を変え、
`tasks` を `cube_lifting` だけに、`limits` を `n_trials: 2` などに
縮めたものです。

## 4. 最初の確認

参照実装を手元のイメージから1本流し、配線を確かめます。HF は通りません。

```bash
export CAPX_HF_TOKEN=hf_xxx          # facebook/sam3 にアクセスできるトークン
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq submit \
    --config ~/capx-queue/queue.yaml \
    --manifest singularity/capx-submission-agent0.yaml \
    --image singularity/images/capx-submission-agent0.sif \
    --submission-id smoke-1
```

- `pending` に `agent0-reference__smoke-1` が入ります。`already queued` と出たら
  `--submission-id` を変えます。手で入れたジョブは `team_id` と `submission_id` で区別します。
- `submit` はイメージをコピーせず、その場所のまま参照します。待機中に作り直すと作り直した方が
  評価されます。

worker を1件だけ処理する設定で投げます。

```bash
mkdir -p logs
env -i PATH="$PATH" HOME="$HOME" LANG="$LANG" CAPX_EVALQ_PYTHON="$CAPX_EVALQ_PYTHON" \
    CAPX_SINGULARITY_MODULE="${CAPX_SINGULARITY_MODULE:-}" \
  sbatch -p <partition> --gres=gpu:1 --cpus-per-task=32 --mem=512G \
    --job-name=capx-smoke -t 06:00:00 --export=ALL \
    scripts/evalq/worker.sbatch --config ~/capx-queue/queue.yaml --max-jobs 1
```

`logs/` は `sbatch` の前に作ります。Slurm はスクリプトを実行する前に出力ファイルを開くので、
無いとジョブが何も残さずに終わります。`env -i` で `sbatch` に渡す環境変数をこれだけにし、
`--export=ALL` で投げるので、`CAPX_HF_TOKEN` などのトークンは worker のジョブに渡りません。
`--export=<変数名>` で名前を挙げる形は使いません。Slurm がノードでログイン時の環境も読み込み
（`--get-user-env` と同じ）、ログイン時のファイルが設定したトークンが入るためです。

実行中のファイルは `runs/<run_id>/<subset>/` にできます。

| ファイル | 中身 |
|---|---|
| `gateway.log` | gateway（審判役） |
| `env-<sim>-<lane>.log` | シミュレータのバックエンド |
| `submission.log` | 提出コンテナ（先頭と末尾の 32 MiB ずつ） |
| `submission-out/` | 提出物が `/capx/out` に書いたもの。参照実装は `logs/vllm.log` を置く |
| `submission-home/` | 提出物の `HOME`（`/capx/home`）。キャッシュなど |
| `gateway-results.json` | gateway の採点結果 |
| `clock-started`・`job-finished` | 提出物が初めて gateway に接続した時刻と、`finish()` を呼んだ時刻（epoch 秒）。`job-finished` ができると runner が提出コンテナを止める |

このディレクトリは、同じチームの次の評価が始まると消えます（運営側の失敗で終わったものは残ります。§1）。
残したいログはその前にコピーします。

終わったら結果を読みます。

```bash
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq status --config ~/capx-queue/queue.yaml
python3 -m json.tool ~/capx-queue/root/results/agent0-reference__smoke-1__*.json
```

見るのは `subsets.<name>.leaderboard.overall_success_rate` です。`cube_lifting` なら
2 trial とも成功するのが普通で、0/2 なら配線か知覚のサーバーを疑います。

次に subset を `taskpacks/original` の Robosuite 7タスクに広げて、新しい `--submission-id` で
同じことをします。参照実装の期待値は成功率 **0.336** です（`cube_lifting` 0.98、`cube_stack` 0.85、
`cube_restack` 0.51、残り4タスクは 0.00〜0.01）。

## 5. HuggingFace からの提出を受け付ける

チームは自分の org のリポジトリに push するだけです（[submission.md](submission.md)）。
何をしたら評価されるかは [PRELIM_RULEBOOK.md](../PRELIM_RULEBOOK.md) §6.4 にあります。

**トークン**: 主催者アカウントで期限の無い fine-grained token を発行し、`CAPX_HF_TOKEN` に入れます。
未設定だと `hf auth login` の期限付きトークンを使い、切れると受け付けがエラーも出さずに止まります。

**チームの登録**: 運営が、チームの org に private の model リポジトリ `capx-submission` を作り、このリポジトリに
`teams/prelim/<org>.yaml`（`team_id: <org>`、`repo: <org>/capx-submission`）を足します（ルールブック §6.1）。
`team_id` は org の名前で固定で、違うファイルは読み込みません。
チームの身元はリポジトリの org で決まり、他チームの org にあるイメージや重みは却下されます。
`tests/evalqueue/test_teams.py` は、`teams/prelim/` のファイルがすべて正しいことも確かめます。

watcher は、確認のたびに `teams.repo` の `teams.ref` を `git fetch` して読み直すので、マージすれば
次の確認から監視が始まります。再起動はいりません。読めなかったファイルは警告を出して飛ばし、
git が失敗したときは前に読んだ一覧を使います。読んだ一覧は `state/teams.json` に残り、
リーダーボードと重みの検査はそれを使います。

```yaml
teams:
  repo: https://github.com/<owner>/<repo>.git
  ref: main
  path: teams/prelim
```

**確認**:

```bash
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq hub-status --config ~/capx-queue/queue.yaml
```

各チームの `revision` と `files` が出れば読めています。`warnings` には次のものが出ます。

| 警告 | 意味 |
|---|---|
| `the token is an expiring one` | 期限付きトークンを使っている。`CAPX_HF_TOKEN` を設定する |
| `invited to org ... but no team is registered for it` | 招待は届いているが、登録がまだマージされていない |
| `team registered under org ... but the account is not a member` | 登録したが招待をまだ受けていない |
| チームの行の `error` | リポジトリが無い、名前の誤り、または権限が無い |

まず1周だけ動かします。

```bash
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq watch --once --config ~/capx-queue/queue.yaml
```

`{"enqueued": ["team-alpha__<key>"]}` のように出れば、イメージは `incoming/` に、
重みは `weights/` に取得されています。

**常駐**: ログインノードの tmux で、ウィンドウごとに次の3つを動かします。

```bash
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq watch --config ~/capx-queue/queue.yaml
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq supervise --config ~/capx-queue/queue.yaml
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq publish --config ~/capx-queue/queue.yaml
```

- `watch` は `hub.poll_interval_s` ごとに各チームの `main` の最新 commit を確認します。
  そのチームの待機中・評価中・評価済みの提出と `submission.yaml` の中身が同じ push は評価しません
  （ルールブック §6.4）。中身は検証した値（既定値のままの項目を除く）を JSON にした sha256 で比べるので、
  コメント・空白・項目の順番は数えません。
  HF が応答しない・429・5xx のときはすぐには却下せず、1時間まで次の確認でやり直します。
  共有ディレクトリの空き不足など主催者側の失敗は却下にせず、ログに出して次の確認でやり直します。
- `supervise` は `slurm` の設定で worker ジョブを `slurm.workers` 本維持し、終わったら投げ直します。
  ジョブ名は `<job_name>-<スロット番号>` で、worker のログは `logs/<job_name>-<n>.<JOBID>.log` です。
  リポジトリのルートで、`logs/` を作ってから動かします。`sbatch` は `PATH`、`HOME`、`LANG`、
  `CAPX_EVALQ_PYTHON`、`CAPX_SINGULARITY_MODULE` だけの環境で、`--export=ALL` で実行します
  （`slurm.WORKER_ENV`）。トークンは worker のジョブに渡りません。`squeue` が失敗したときは、その回は投げません。
- `publish` は `leaderboard.json` を HF の静的 Space に push します（§5.1）。
- トークンを持つ `watch` は worker と同じジョブで動かしません。`scripts/evalq/queue-test.sbatch` は
  2つを1つのジョブで動かしますが、スモークテスト用です。

`incoming/<team_id>/<key>/` は、そのジョブの結果が出ると worker が消します（HF にイメージが残るため）。
キューのどの記録（待機中・評価中・終了）からも指されていないものは、watcher がそのチームを確認するたびに消します
（チームの最新の提出で、まだ取り込んでいる途中のものは残す）。却下・置き換え・HF や共有ディレクトリの障害など、
キューに入らずに終わった理由は問いません。締切の後は全チームについて消します。
イメージは提出ごとに取得するので、重みだけを変えた提出でもイメージを取り直します。
運営側の失敗（`internal_error`、`no_score`）で終わったときは残すので、そのイメージを `submit` で
入れ直します。それ以外で同じ提出をもう一度評価したいときは、チームに `submission_id` を変えて push してもらうか、
HF から取得して `submit` します。

### 5.1 リーダーボード

`evalq publish` は `results/` とキューの状態から `leaderboard.json` を作り、HF の静的な Space に
push します。push するのはこの JSON だけです。ページ（日本語版 `web/leaderboard/index_jp.html` を
Space に `index.html` として、英語版 `index_en.html` はそのままの名前で置く）は、ページを変えたときに
手でアップロードします。URL は `https://<owner>-<space>.static.hf.space` です。

```yaml
leaderboard:
  repo: <owner>/<space>                # 空なら公開しない
  repo_type: space
  token_env: CAPX_HF_LEADERBOARD_TOKEN # この repo だけに書ける fine-grained トークン
  interval_s: 60       # 60 未満は不可（HF は1 repo あたり1時間128 commit まで）
  heartbeat_s: 1800    # 変化が無くてもこの間隔で push（ページが「止まった」と見分けるため）
  subset_labels: {}    # subset 名 → ページに出す名前
```

```bash
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq publish --dry-run --config ~/capx-queue/queue.yaml  # 中身の確認
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq publish --once    --config ~/capx-queue/queue.yaml  # 1回だけ
```

- **順位**: 各チームの一番良い提出。全 subset に数字がある提出だけが順位に入ります。スコア（パックの
  `score_weights` で重みを付けた、シミュレータごとの成功率の平均。無ければ全試行の成功率）が同じなら
  Time Taken が短い方が上で、どちらも同じチームは同順位です。同順位のチームは、先に終わった方を上に並べます。
- **Time Taken**: subset ごとのジョブの時計（提出物が初めて gateway に接続してから、`finish()` を呼ぶか終了するまで）を、
  その subset のジョブ全体の時間で頭打ちにした合計です。subtitle の上限（全 subset の合計）を超えません。
  履歴の Wall Time は起動を含めた実際の経過時間です。
- **over-time**: ジョブ全体の時間を使い切った subset です。記録された分で採点し、順位にも入ります。
  時間切れでも exit 0 で終わる提出が多いので、runner が gateway の時計（`clock-started`・`job-finished`）
  と提出物が止まった時刻から判定して `SubsetResult.over_time` に書きます。提出物の出力は見ません。
- **出さないもの**: subset・タスクごとのスコア、タスク数、subset ごとの時間、エラーの文面、repo、パス。
  これらは `results/`、`queue/done|failed/`、`state/`、`runs/`（各チームの最新の評価と運営側の失敗）に残ります。
  `evalq archive` はこれらを1つの `backlog-<時刻>.tar.gz` にまとめます。
- 中身が変わったときだけ push し、`heartbeat_s` ごとには変化が無くても push します。ページは
  15 秒ごとに読み直し、`generated_at` が古くなると「クラスタから報告が無い」と出します。
- 登録されたチーム（`state/teams.json`）だけが載ります。`publish` は CPU だけの別ジョブ
  （`scripts/evalq/leaderboard-publisher.sbatch`）で動かせば、評価を止めずに再起動できます。
- 本番への切り替えのチェックリストは、運営の非公開リポジトリ（`private/docs/`）にあります。

## 6. BEHAVIOR を別ホストの Docker で動かす

Isaac Sim は RT コアの無い GPU では動きません。そこで BEHAVIOR のバックエンドだけを、RT コアのある
GPU を持つ別ホストで Docker として動かします。gateway と提出コンテナはこれまでどおり計算ノードです。
worker が ssh でコンテナを起動し、そのソケットを `ssh -L` で計算ノードの `be/` へ転送するので、
gateway からはローカルのバックエンドと同じに見え、提出コンテナからは見えません。

**別ホストの準備**（Docker、NVIDIA Container Toolkit、rsync が必要）:

```bash
git clone https://github.com/matsuolab/geniac_prize_2026_code.git cap-x-geniac && cd cap-x-geniac
docker compose build env-behavior                                 # capx-env-behavior:local
docker compose --profile assets run --rm fetch-behavior-assets    # アセット約37GB
docker volume ls | grep behavior                                  # ボリューム名を確認
```

`capx/` を変えたら `docker compose build env-behavior` をやり直します。

**ssh**: worker は計算ノードから `ssh -o BatchMode=yes <behavior-host>` で接続し、パックを rsync で
送ります。パスフレーズの無い鍵を計算ノードから使える `~/.ssh/config` に書き、確かめます。

```bash
sbatch -p <partition> -t 00:10:00 -o logs/ssh-check.%j.log \
    --wrap 'ssh -o BatchMode=yes <behavior-host> "docker images capx-env-behavior"; echo rc=$?'
```

**設定**: そのシミュレータに `image` の代わりに `remote` を書きます（両方書くと検証で落ちます）。

```yaml
subsets:
  - name: original
    taskpack: taskpacks/original
    simulators: [robosuite, libero, behavior]
simulators:
  robosuite: {image: singularity/images/capx-env-robosuite.sif, max_lanes: 32}
  libero: {image: singularity/images/capx-env-libero.sif, max_lanes: 32}
  behavior:
    max_lanes: 8                     # 1 レーン約 9 GiB。96 GB の GPU 1枚ならこれが上限
    remote:
      host: <behavior-host>          # 計算ノードの ssh config での名前
      image: capx-env-behavior:local
      gpu: "0"                       # docker run --gpus device=<gpu>。GPU の番号を1つ。評価はこの GPU ごとに順番に使う
      volumes:
        - cap-x-geniac_behavior-assets:/data
        - cap-x-geniac_behavior-appdata:/cache
```

コンテナは `--network none` で作り（`docker create`、`docker start -a`）、終わると worker が `docker rm -f` で消します。
レーンごとに1コンテナが同じ GPU に載ります。1つあたり GPU メモリを約 9 GiB、ホストのメモリを約 5 GiB 使い、
準備ができるまで約 3〜4 分かかります（RTX PRO 6000 Blackwell で測定）。96 GB の GPU 1枚に載るのは8レーン
（約 72 GiB）までなので、`max_lanes` は 8 です。ログは `runs/.../<subset>/` の `env-behavior-<lane>.log`（コンテナの出力）と
`forward-behavior-<lane>.log`（ソケット転送）です。最初のエピソードは環境の構築に数分かかります。

**GPU は一度に1つの評価だけが使います**: 2つの評価の BEHAVIOR は1枚の GPU に載りません。

- 待機中のジョブがあると、worker はジョブを取る前に、使う別ホストの GPU ごとにロック（別ホストの
  `<socket_dir>/capx-gpu-<gpu>.lock` への `flock`）を取ります。ロックは ssh を1本つないだまま持ち、
  ジョブの結果を出すかキューに戻すまで（片付けも含めて）放しません。worker が強制終了されると ssh が切れ、
  ロックは外れます。バックエンドは、その GPU のロックを持っている間しか起動しません。
- 評価の途中でロックが外れた（ssh が切れた）ときは、ほかの評価がその GPU で始まり、この評価のコンテナを
  消しているかもしれません。そこで評価を止め、結果を出さずにジョブをキューに戻します。worker が落ちたときと同じく
  1回と数えます。`scancel` で worker を止めたときもロックの ssh は切れますが、これは普通の停止として扱い、数えません。
- ロックを取るたびに、別ホストの `<socket_dir>/capx-gpu-<gpu>.gen` にある数（世代）を1つ増やします（一時ファイルに
  書いてから `mv` で置き換えます）。その GPU のコンテナやソケットのディレクトリを消したり作ったりするコマンド
  （残ったコンテナの片付けと、バックエンドの `docker create`）は、別ホストの上でこの数が自分の世代と同じかを
  確かめてから動きます。ssh が遅れて、届くまでの間に別の worker がロックを取っていれば、何もせずに終わり、
  ロックが外れたときと同じく評価を止めます。
- 世代を増やすことと、確かめてから消す・作るまでは、別ホストの `<socket_dir>/capx-gpu-<gpu>.op` の短いロックの中で
  行います（待つのは60秒まで）。そのため、確かめたあとで `docker` が遅れても、その間に新しい世代はできません。
  バックエンドはこのロックの中で `docker create` し、ロックの外で `docker start -a` します。古い世代のうちに作られた
  コンテナは新しい世代の片付けが消し、新しい世代ができたあとの `docker create` は断られます。
- 結果を出す前に、別ホストの世代がまだ自分の世代かを確かめます。ネットワークが切れている間に、別ホストの側で
  ロックが外れてほかの worker が取ることがあり、そのあいだ手元の ssh はまだ切れていないためです。違うときと、
  確かめられない（ssh が失敗した）ときは結果を出さず、ロックが外れたときと同じくジョブをキューに戻します。
- ロックを取れない（同じ GPU で別の評価が動いている）ときと、別ホストが答えないときはジョブを取らず、
  キューが空のときと同じ間隔（`watcher.poll_interval_s`）で取り直します。待機中のジョブはそのまま待ち、
  SIGTERM（`scancel`）ではいつもどおり止まります。同じホストの別の GPU を使うキューどうしは待ちません。
- 同じ GPU を使うキュー（中間評価と順位評価など）は、同じ `remote.socket_dir` にします。ロックファイルがそこにあるためです。
  `remote.socket_dir` は絶対パスで書き、設定を読むときに `/tmp/` や `//tmp` は `/tmp` にそろえます。`..` は
  使えません（別ホストではシンボリックリンクをたどってから `..` を解くので、ここでそろえられないため）。
- `remote.gpu` は GPU の番号（`0`、`1` など）を1つ、先頭に `0` を付けずに書きます。`all`、UUID、`00`、`01` は
  設定の検証で落ちます。ロックファイルの名前が番号なので、同じホストを使うキューはどれも GPU を番号で書きます。
- `remote` のシミュレータを使う subset があるキューは `slurm.workers: 1` にします。2本目の worker は待つだけなので、
  2以上は設定の検証で落ちます。`simulators` を書かない subset は、パックに BEHAVIOR のタスクが無くても使うものとして
  数えます（設定を読むときにはパックを見ないため）。
- 評価を動かしている間は、その GPU をほかの人の処理に使わせません。ほかのプロセスがメモリを使うと8レーンが載りません。

待っている間、worker ログには10分に1回 `not claiming jobs until the remote GPU is free: <理由>` が出ます。
空くと `the remote GPU is free again` が出て、ジョブを取り始めます。評価中にロックの ssh が切れると
`lost the lease on <host> GPU <gpu>` が出て、上のとおり評価が止まります。ネットワークが切れたとき、別ホストの側で
ロックが外れるのは sshd が切断に気づいたときです（別ホストの sshd に `ClientAliveInterval` を設定しておくと
早く気づきます）。

**残ったコンテナ**: worker が強制終了される（Slurm の KillWait、OOM、ノードの障害）と、別ホストのコンテナは
残って GPU メモリを使い続けます。そこで各コンテナに使う GPU のラベル `capx.gpu=<gpu>` を付け、別ホストの
ソケットのディレクトリ名も `<socket_dir>/capx-<ランダム>-gpu-<gpu>` にします。ロックを持つ worker のほかに
その GPU で評価しているものは無いので、worker は subset ごとに gateway を起動する前に、その GPU のラベルが付いた
コンテナとディレクトリを、どのキューのものでもすべて消します。ラベルの無いコンテナと、ほかの GPU のものには
触れません。消せなかったときは、提出物を起動せずにその subset を `internal_error` で終えます。

別ホストのソケットのパスは `-gpu-<gpu>` のぶん長くなります。AF_UNIX の 107 バイトを超えないかを、gateway を
起動する前に一番大きいレーン番号のパスで確かめ、超えるときはその subset を `internal_error` で終えます。

## 7. 日常の操作

```bash
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq status --config ~/capx-queue/queue.yaml
```

`pending`、`claimed`、`done`・`failed` の件数と、最近の結果（`team_id`、`status`、`error`）が出ます。
`error` はパスや例外の文面を含むので主催者向けです。

**worker が落ちたとき**: worker は評価中のジョブの claim を5分ごとに更新します。Slurm に止められる
（`scancel`、時間切れ）と、Slurm は提出コンテナにも SIGTERM を送るので、その評価は提出物について
何も言えません。worker は評価中の提出を止め、結果を出さずにジョブをキューへ戻して終了します。
次の worker が最初から評価し直します。強制終了されたときは claim が `claimed` に残り、更新が15分止まると
同じスロットの worker が（別のジョブを評価中でも）戻します。それ以外は手で戻します。

- worker が落ちた評価が3回を超えたジョブは、評価せず `internal_error` で終えます。別ホストの GPU のロックが外れて止めた評価も数えます（§6）。
- 結果を出したあとで worker が落ち、claim だけが残ったジョブは、評価し直さずに `done`・`failed` へ移します。

```bash
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq requeue-stale --config ~/capx-queue/queue.yaml --dry-run
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq requeue-stale --config ~/capx-queue/queue.yaml   # 既定 --older-than-h 1
```

どのジョブも使っていない重みは、worker がジョブを終えるたびに消します。

**締切と順位評価**: 中間評価の設定に `deadline` を書いておくと、その時刻に watcher が各チームの `main` の
commit を記録します（`<root>/state/ranking/heads/deadline.json`）。締切までに作られた最新の commit で、
取得中のイメージがあっても待たず、チームの登録も読み直しません。各チームの `main` は同時に読み、
読めなかったチームと30秒以内に答えがなかったチームは、watcher が締切の前に最後に見た commit を使います
（`heads/seen.json`）。そのあと、記録した commit の `submission_ranking.yaml` を検査して記録します
（`<root>/state/ranking/<team_id>.json`、全チームが済むと `FINAL`）。イメージや重みが読めないチームは
次の確認で同じ commit をやり直し、締切から1時間たっても読めなければ却下として記録します
（存在しない commit を指していると、いつまでも読めないため）。

締切の後、順位評価用の設定（別の `root` と順位評価のパック）で、記録したものをキューに入れて worker を動かします。
中間評価の worker と同時に動かしても、BEHAVIOR の GPU は一度に1つの評価だけが使います（§6）。
順位評価用の設定には `deadline` と `leaderboard.repo` を書かず、`master_seed` は中間評価と別の値にします。
`deadline` を残すと worker が入れた評価をすぐに取り消し、`leaderboard.repo` を残して `publish` すると、
順位評価の結果が公開ページに出ます。

```bash
$CAPX_EVALQ_PYTHON -m capx.cli.main evalq queue-ranking --config <順位評価の設定> \
    --interim-root <中間評価の root>
```

1チーム1回だけ入ります。却下されていたチームには `rejected` の結果を出し、出さなかったチームは
飛ばします。運営側の障害で止まった評価だけ、`--team <team_id> --again` でもう一度入れます。

**ログの場所**

| どこ | 何 |
|---|---|
| `watch`・`supervise`・`publish` を動かしている tmux | 取得・却下・投入、worker の投入 |
| `logs/<job_name>-<n>.<JOBID>.log` | worker（どのジョブを評価し、どう終わったか） |
| `<root>/runs/<run_id>/<subset>/` | 1回の評価の各コンテナのログ（§4）。各チームの最新の評価と、運営側の失敗で終わった評価の分が残る |
| `<root>/results/<run_id>.json` | 結果。`error` と `error_code` に失敗の理由 |

## 8. 別のクラスタへ移す

コードと既定値はリポジトリにあるので、移すのは共有ディレクトリの一部、設定ファイル、secrets です。

1. **止める**: `watch` を止め、`status` の `pending` と `claimed` が空になるまで待ちます。
   `supervise` を止め、`squeue -u $USER` で worker が残っていないことを確かめてから `publish` を止めます。
   `queue/` には旧クラスタの絶対パスが入っているので、空にしてから移ります。
2. **持っていく**: ラウンドごとの `root` について、次のとおりにします。

   | パス | 扱い |
   |---|---|
   | `results/` | 必ず移す。成績そのもので、公開ページもここから作る |
   | `state/` | 必ず移す。無いと全チームの最新の提出がもう一度評価される |
   | `runs/` | 必要なら（ログ） |
   | `incoming/`、`weights/` | 移さない。次の提出で HF から取り直す |
   | `queue/` | 移さない |

3. **作り直す**: 新しいクラスタで §2 に沿って venv、submodule、イメージを用意します
   （イメージは `.sif` を運んでも構いません）。クラスタごとに違うのは次の点です。
   - パーティション名 → `slurm.partition`
   - Singularity のモジュール名 → `CAPX_SINGULARITY_MODULE`
   - 共有ディレクトリ → `root`（計算ノードから見える場所）
   - ノードの GPU・メモリ → `slurm.mem`、`slurm.cpus_per_task`、`slurm.workers`
   - BEHAVIOR のホスト → 計算ノードからの ssh を確かめ直す
4. **secrets を設定し直す**: `CAPX_HF_TOKEN`、`CAPX_HF_LEADERBOARD_TOKEN`。
5. **再開する**: `verify-isolation.sbatch` を流し、`hub-status` で全チームが読めることを確かめ、
   参照実装を1本流してから `watch`・`supervise`・`publish` を動かします。公開ページに移す前と同じ
   履歴が出ていることを確かめてから、チームに再開を伝えます。

## 9. うまくいかないとき

| 症状 | 原因と対処 |
|---|---|
| 設定の読み込みが `extra_forbidden` で失敗 | 存在しない項目を書いている。§3 の項目と照らし合わせる |
| sbatch のジョブが何も残さずすぐ終わる | `logs/` が無い。リポジトリのルートで `mkdir -p logs` |
| `submission image missing` / `gateway image missing` / `env-backend image missing` | イメージのパスが違うか、ビルドに失敗している。`capx-sif-build.<JOBID>.log` を見る |
| `unix socket path is ... over the 107-byte kernel limit` | `socket_dir`（`<sim> backends:` と付いていれば別ホストの `remote.socket_dir`）が長い。短いパスにする |
| `submission did not reach the gateway` | 提出物が `submission_startup_s` 以内に接続しなかった。採点せず `failed`（`submission_startup_timeout`）。`submission.log` を見る |
| `submission wrote more than ...` / `could not measure what the submission wrote` | 提出物が `/capx/out` と `HOME` に上限を超えて書いたか、読めないディレクトリか深すぎるディレクトリを作ったか、記述子を読めないプロセスがある（§1）。それまでの分で採点し `failed`（`submission_storage_exceeded`）。`submission-out/` と `submission-home/` を見る |
| 参照実装で vLLM が起動しない | `submission-out/logs/vllm.log` を見る。既定の `--attention-backend TRITON_ATTN` を外さない |
| エージェントは動くが何も掴めない | SAM3 か GraspNet の重みが無い。`submission.env` の `CAPX_SAM3_CHECKPOINT`、`submission.log` の `/models/sam3` の bind、GraspNet の `model.pt` が実体かを確かめる |
| `has not been granted access to gated repo` | 主催者アカウントがそのリポジトリのアクセスを申請し、承認後にもう一度提出してもらう |
| `declared weights are not in ...` | 受け付けたあとに重みが消された。`weights/` を手で消していないか確かめ、もう一度入れる |
| push されたのにキューに入らない | `hub-status` の警告と `revision` を見る。トークン切れ、未登録、`submission.yaml` が前の提出と同じ（イメージや重みだけを push した）のどれか |
| worker ログに `could not start the behavior-<n> backend` | そのエピソードは無効になる。`did not create <host>:...` なら別ホストの `env-behavior-<n>.log` でイメージ名・ボリューム名・GPU 番号を、`exited with code 255` なら計算ノードからの ssh を確かめる |
| worker ログに `not claiming jobs until the remote GPU is free: <host> GPU <gpu> is in use by another evaluation` | 同じ GPU で別の worker（同じキューの2本目、ほかのキュー、スモークテストなど）が評価している。終われば次の確認でジョブを取る。ずっと続くなら、別ホストで `lslocks \| grep capx-gpu` を見てロックを持っているプロセスを確かめる |
| worker ログに `not claiming jobs until the remote GPU is free: could not take the lease on <host> GPU <gpu>`、または `<host> did not answer` | 計算ノードから別ホストへ ssh できないか、別ホストに `flock` が無い。§6 の ssh の確認をやり直す |
| worker ログに `lost the lease on <host> GPU <gpu>`、または `<host> GPU <gpu> has been taken over since lease <n>` | 評価中にロックの ssh が切れた（後者は、遅れて届いたコマンドを別ホストが断った）。その評価は止まり、結果を出さずにジョブがキューに戻る（1回と数え、3回を超えると `internal_error`）。計算ノードと別ホストの間の接続を確かめる。`scancel` で worker を止めたときにも出るが、そのときは数えない |
| `could not remove the leftover containers on <host>` | 別ホストへの ssh か Docker が動いていない。`exit 75` なら `capx-gpu-<gpu>.op` のロックが60秒空かなかった。その subset は `internal_error` で終わるので、直してからイメージを `submit` で入れ直す |
| 結果の代わりに worker ログに `could not confirm the lease on <host> GPU <gpu>` | 結果を出す前に、別ホストで世代を確かめられなかった。結果は出さず、ジョブはキューに戻る（1回と数える）。計算ノードから別ホストへの ssh を確かめる |
| `gateway.log` に `the <sim> backend at <socket> stopped answering` | そのバックエンドとの接続が切れたか、30 分応答しなかった。その試行は `env_crash` で無効になり、応答が戻るかプロセスが終わるまで、そのバックエンドでは次の env を作らない（そのシミュレータのレーンが1つ減る）。`env-<sim>-<lane>.log` を見る。固まったままならそのプロセスを止める。runner が次に要求されたときに起動し直す |
| ジョブが `claimed` のまま進まない | worker が落ちている。`squeue` で確かめ、`requeue-stale` |
