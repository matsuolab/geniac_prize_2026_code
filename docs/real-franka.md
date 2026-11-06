# 実機Franka Panda Quick Start

[公式ドキュメント](https://github.com/capgym/cap-x/blob/main/docs/real-franka.md)

CaP-Xから実機のFranka Pandaを動かすための手順です。

事前に
[robots_realtime](https://github.com/uynitsuj/robots_realtime.git)
をクローンし、同リポジトリの手順に従ってFrankaを起動できることを確認してください。

例えば、RobotiqグリッパをRS485で接続している構成では、以下の設定を使用して動作確認できます。

```text
configs/franka/franka_robotiq_viser_teleop.yaml
```

CaP-X側の既定設定は以下です。

```text
env_configs/real/real.yaml
```

既定タスクは`"pick up the red cube and lift it"`です。別のタスクを実行する場合は、同ファイル内の`task_only_prompt`と`prompt`を変更してください。

例えば、以下のようなタスクも指定できます。

```text
stack these objects as high as possible
```

## 必要な機材

Franka Pandaに加えて、キャリブレーション済みのDepthを取得できるステレオカメラが必要です。

例えば、ZEDステレオカメラを利用できます。

Depth画像は、SAM3とContact-GraspNetから3次元把持姿勢を生成するために使用されます。そのため、RGB画像のみを出力する単眼カメラでは、この構成をそのまま利用できない場合があります。

## Step 1: カメラ外部パラメータを作成する

`robots_realtime`では、ロボットのワールド座標系におけるカメラ姿勢を、外部パラメータファイルから読み込みます。

環境に合わせたYAMLファイルを、以下に作成します。

```text
robots_realtime/configs/camera_extrinsics/
```

既存のAutoLab ZED設定は、ファイル形式の参考として利用できます。

```text
configs/camera_extrinsics/autolab_franka_zed_top.yaml
```

例えば、以下のようにカメラ位置と姿勢を設定します。

```yaml
# ロボットのワールド座標系を基準としたカメラ外部パラメータ
#
# position: [x, y, z]、単位はm
# rpy_radians: [roll, pitch, yaw]、単位はrad

position: [1.007, 0.0, 0.29]
rpy_radians: [1.0472, 3.14159, -1.5708]
```

この値は既存環境の例です。実際には、自分の環境におけるカメラの設置位置と姿勢へ変更してください。

## Step 2: クライアント設定から外部パラメータを参照する

使用する`robots_realtime`のFrankaクライアント設定を開きます。

例えば、以下の設定です。

```text
robots_realtime/configs/franka/franka_robotiq_client.yaml
```

`ZedCamera`センサーノードの`extrinsics_file`を、Step 1で作成したファイルへ変更します。

```yaml
extrinsics_file: "configs/camera_extrinsics/your_setup.yaml"
```

外部パラメータが不正確な場合、認識した物体位置と実際の把持位置にずれが生じます。CaP-Xを実行する前に、カメラとロボット間の位置・姿勢をキャリブレーションしてください。

## Step 3: CaP-Xを起動する

CaP-Xリポジトリで、Contact-GraspNetを含む依存関係をインストールします。

```bash
uv sync --active --extra contactgraspnet
```

続いて、実機用設定を指定してCaP-Xを起動します。

```bash
uv run --no-sync --active \
    capx/envs/launch.py \
    --config-path env_configs/real/real.yaml
```

起動後、表示されたURLへブラウザからアクセスします。URLにはアクセス用のトークンが付いていて、起動のたびに変わります。

```text
http://127.0.0.1:8200/#token=...
```

Web UIは既定でこのマシン（127.0.0.1）でだけ待ち受けます。別のマシンのブラウザから使うときは、`ssh -L 127.0.0.1:8200:127.0.0.1:8200 <host>` でポートを転送してください（`localhost` は IPv6 の ::1 になることがあり、そこで別のユーザーが待ち受けているとトークンが漏れるので、`127.0.0.1` と書きます）。どうしても他のアドレスで待ち受けたいときは `CAPX_WEB_UI_HOST=0.0.0.0` を付けて起動します。そのときは、ポートに届きトークンを知っている人なら誰でも、このマシンでコードを実行できます。

CaP-X側に以下の表示が出た場合、実機環境からの観測を待機しています。

```text
Waiting for observation from real environment...
Waiting for observation from real environment...
```

CaP-Xは`127.0.0.1:9000`で`robots_realtime`クライアントからの接続を待ち受けるため、既定では同じマシン上のクライアントだけが接続できます。クライアントを別のマシンで動かす場合は、`env_configs/real/real.yaml`の`low_level`を次のように書き換えて、待ち受けるアドレスを指定してください。このポートに接続できる相手は誰でもロボットを操作できるため、信頼できるネットワーク内でのみ公開してください。

```yaml
    low_level:
      _target_: capx.envs.simulators.franka_real.FrankaRealLowLevel
      host: 0.0.0.0
      port: 9000
```

## Step 4: `robots_realtime`クライアントを起動する

別のターミナルを開き、`robots_realtime`リポジトリへ移動します。

使用するFrankaクライアント設定を指定して、セッションを起動します。

```bash
uv run rr-session \
    configs/franka/franka_robotiq_client.yaml
```

クライアントが接続されると、カメラ観測とロボット状態がCaP-Xへ送信され、Web UIからタスクを実行できるようになります。
