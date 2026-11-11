# 提出の手順

手元で評価してから提出するまでの手順です。提出物が満たすこと、何をしたら評価されるか、
採点と順位は [PRELIM_RULEBOOK.md](../PRELIM_RULEBOOK.md) にあります。

## 1. 手元で評価する

公開パック `taskpacks/original`（Robosuite 7、LIBERO 130、BEHAVIOR 50 タスク）は、
配布したサーバーで自分たちで評価します。Robosuite と LIBERO は B300 のサーバーで、
BEHAVIOR（Isaac Sim）は RTX PRO 6000 のサーバーで動かします。

まず Robosuite だけを、インプロセスで動かして配線を確かめます。

```bash
capx eval --taskpack taskpacks/original --mode dev --simulators robosuite
```

本番と同じように、gateway と各シミュレータを別プロセスにして全タスクを動かす方法は
[docker/README.md](../docker/README.md) にあります。BEHAVIOR の環境の用意は
[behavior-tasks.md](behavior-tasks.md) を見てください。

## 2. イメージを作る

提出物は Singularity のイメージ（`.sif`）です。参照実装（CaP-Agent0）の定義が
`singularity/capx-submission-agent0.def` にあるので、これを元に作れます。

本番の実行環境（ルールブック §5）と同じ条件にして確かめてください。

- ネットワークはありません。モデルの重みは、イメージに入れるか、`submission.yaml` の `weights` でマウントします。
  どちらの場合も、評価中に使うモデルはルールブック §7.2 のとおり申告します。
- gateway には `CAPX_ENV_SOCKET` のソケットで接続します。
- 書き込めるのは `CAPX_OUTPUT_DIR`（`/capx/out`）と `HOME` の下、それに `/tmp` と `/var/tmp`（合わせて 64 MiB）だけです。
- `CAPX_OUTPUT_DIR` と `HOME` に書けるのは、合わせて 20 GiB、ファイルとディレクトリを合わせて 100,000 個までです。
  超えると評価の途中で止められます。

参照実装を元にするときは、次の点に注意してください。

- vLLM が配信するモデルの名前とポートは `CAPX_MODEL` と `CAPX_SERVE_PORT` で決まり、
  エージェントもそのモデルとポートに問い合わせます（`runner.yaml` の `model` と
  `server_url` より優先します）。
- `runner.yaml` とタスクごとの設定（`env_configs/gateway/*.yaml`）に知らないキーがあると、
  起動したときにエラーになります。`task_map` の `config` を相対パスで書くと、
  `runner.yaml` のあるディレクトリからの相対になります。
- カメラの画像を使うかどうかは `use_visual_feedback`、`use_img_differencing`、
  `use_video_differencing` だけで決まります。画像を受け付けないモデルでは `false` にします。
- gateway が `env_crash` で無効にした試行は返却されるので、ランナーはその試行を
  もう一度実行します（2回まで）。gateway との接続が切れたレーンは止まり、その試行は
  ほかのレーンが実行します。試行が残ったままレーンがなくなったときは、`finish()` を
  呼ばずにエラーで終了します。

## 3. 提出用リポジトリを用意する

1. チームの org に、提出用の model リポジトリを1つ作ります（例: `team-alpha-org/capx-submission`）。
2. このリポジトリに `teams/prelim/<team_id>.yaml` を足す pull request を送ります（ルールブック §6.1）。

## 4. 提出する

イメージを先に push し、その commit の SHA を `submission.yaml` に書いてから push します。
`submission.yaml` には、自分自身が入る commit の SHA は書けないためです。

```bash
# 1. イメージを push して commit の SHA を得る
python -c "from huggingface_hub import HfApi; print(HfApi().upload_file(
    path_or_fileobj='agent.sif', path_in_repo='agent.sif',
    repo_id='team-alpha-org/capx-images', commit_message='week 3 image').oid)"

# 2. その SHA を submission.yaml の image.revision に書いて push する
hf upload team-alpha-org/capx-submission submission.yaml submission.yaml --commit-message "week 3"
```

評価されるのは、`submission.yaml` の中身が前の提出と違うときだけです（ルールブック §6.4）。
新しいイメージや重みを push したら、その commit の SHA を `image.revision` や `weights[].revision` に書いて
`submission.yaml` を push します。同じものをもう一度評価してほしいときは、`submission_id` を変えます。

順位評価には、同じ形式の `submission_ranking.yaml` を同じリポジトリに push します。

結果は運営が案内するリーダーボードに出ます。提出ごとのステータスは done / running / waiting / deferred /
cancelled / over-time / rejected / failed で、却下（rejected）や失敗（failed・over-time）にはその理由の
エラーコードが番号（E01 など）で横に出ます。エラーの文面やログはリーダーボードには出ません。
評価中と終了した提出には、全ベンチマークを合わせた試行のうち実行済みの割合が整数の % で出ます（切り捨て）。試行の数そのものは出ません。

| コード | エラーコード | 意味 |
|---|---|---|
| E01 | `invalid_manifest` | submission.yaml が無い、または不正 |
| E02 | `image_not_found` | 指定した revision にイメージが無い |
| E03 | `image_too_large` | イメージがサイズ上限を超えている |
| E04 | `image_hash_mismatch` | イメージが sha256 と一致しない |
| E05 | `foreign_repo` | イメージか重みがチーム自身の org に無い |
| E06 | `repo_unreadable` | リポジトリを読めなかった（アクセス権を確認） |
| E07 | `weights_gated` | 運営がアクセスできない gated な重み |
| E08 | `weights_too_large` | 重みがサイズ上限を超えている |
| E09 | `submission_failed` | 提出物がエラーで終了した |
| E10 | `submission_timeout` | ベンチマークで時間を使い切った |
| E11 | `submission_startup_timeout` | 起動後、時間内にベンチマークへ接続しなかった |
| E12 | `submission_storage_exceeded` | 書き込める量の上限を超えた |
| E13 | `no_score` | スコアが出なかった |
| E14 | `internal_error` | 運営側の問題（運営が確認します） |
