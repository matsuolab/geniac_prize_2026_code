# BEHAVIOR タスク

[公式ドキュメント](https://github.com/capgym/cap-x/blob/main/docs/behavior-tasks.md)

[BEHAVIOR-1K](https://behavior.stanford.edu/) は、日常生活における1,000以上のタスクを定義したベンチマークです。CaP-Xでは、NVIDIA Isaac SimとOmniGibson上でR1Proヒューマノイドを動かし、BEHAVIOR-1Kのタスクを評価します。

Dockerイメージのビルド、コンテナ起動、BEHAVIOR環境のセットアップ、スモークテストは[docker/README.md](../docker/README.md)の「開発用コンテナ」を参照してください。以下では、スモークテスト後にタスクを実行する流れを説明します。

## Step 1: BEHAVIOR用venvを有効にする

コンテナ内で、BEHAVIOR用のvenvと実行時環境を有効にします。

```bash
cd /workspace/cap-x
source capx/third_party/b1k/.venv/bin/activate

export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_GPU_ID=0
export PYTHONPATH="$PWD:$PYTHONPATH"
```

コンテナにはGPUが1枚だけ見えるため、OmniGibsonから見たGPU番号は`0`です。

## Step 2: タスクを実行する

### oracleコードを実行する

oracleは、LLMにコードを生成させる代わりに、タスククラスにあらかじめ定義された参照Pythonコードを実行するモードです。物体位置を正解データとして直接与えるものではありません。例えばradioタスクでは、参照コードがSAM3とRGB-D画像から物体を検出し、深度から3次元位置を推定したうえで、把持姿勢生成・移動・IK・グリッパ操作のAPIを呼び出します。

環境とAPIサーバーの接続確認には、oracle設定を使います。

```bash
python capx/envs/launch.py --config-path env_configs/r1pro/r1pro_pick_up_radio_oracle.yaml --total-trials 1
```

### LLMでタスクを評価する

LLM評価では、先にOpenAI互換プロキシを同じコンテナ内で起動します。別のターミナルで実行してください。

```bash
python capx/serving/openai_server.py   # model.sock で待ち受ける
```

その後、タスク設定とモデルを指定します。

```bash
python capx/envs/launch.py --config-path env_configs/r1pro/b1k_hiding_Easter_eggs.yaml --model gpt-5 --total-trials 1
```

> **コンペでの注意:** コンペで使えるモデルは、[ルールブック §7](../PRELIM_RULEBOOK.md#7-使えるモデル) の条件を満たすものだけです。GPT・Claude・Gemini などのクローズドモデルは、評価にもデータ生成にも使えません（§7.1）。この例のモデルは元の CaP-X のものです。

この`launch.py`コマンドでは、次の処理が順番に行われます。

1. 指定したYAMLを読み込み、環境・試行回数・出力先を決める。
2. YAMLの`api_servers`に定義されたSAM3（`sam3.sock`）とContactGraspNet（`graspnet.sock`）を自動起動する。そのソケットですでにサーバーが待ち受けていれば、既存のサーバーを利用する。
3. `R1ProBehaviorCodeEnv`と、低レベル環境の`R1ProBehaviourLowLevel`（OmniGibson / Isaac Sim）を初期化する。
4. 指定した`gpt-5`へタスクのプロンプトを送り、実行用のPythonコードを生成する。
5. 生成コードから`R1ProControlApi`の関数を呼び出し、SAM3による物体検出、ContactGraspNetによる把持姿勢生成、PyRoKiによるIK、移動・グリッパ操作を実行する。
6. 実行後のシミュレータ状態から報酬とタスク完了状態を計算し、ログ・生成コード・動画を`outputs/`へ保存する。

multi-turn設定を持つYAMLでは、コード実行後にstdout・stderr・画像差分をLLMへ返し、`REGENERATE`（コード再生成）または`FINISH`（終了）を判断させます。上記の`b1k_hiding_Easter_eggs.yaml`はこの設定を持たないため、1回のコード生成・実行で評価します。

#### ローカルQwenサーバーを使う

APIキーを使わず、ローカルのQwen3.6-27Bを使う場合は、同じコンテナ内でQwenサーバーを起動します。

```bash
bash /workspace/docker/run-qwen-server.sh &
```

サーバーの起動後、次のように評価します。

```bash
source capx/third_party/b1k/.venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

OMNI_KIT_ACCEPT_EULA=YES \
OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
    --model "Qwen/Qwen3.6-27B" \
    --server-url "http+unix://model.sock/v1/chat/completions" \
    --max-tokens 8192 \
    --total-trials 1 \
    --num-workers 1 \
    --record-video True
```

`run-qwen-server.sh`はコンテナ内のUnixソケット`model.sock`でQwenのOpenAI互換APIを提供します。初回起動時はモデル重みのダウンロードに時間がかかる場合があります。Qwenサーバーのエンドポイントは`/v1/chat/completions`のため、この場合は`--server-url`の指定が必要です。

## タスク設定

BEHAVIOR-1KのCaP-X対応タスクは、B10、B20、B30、B40、B50の5区分、合計50タスクです。設定ファイルは次のように分かれています。

| 範囲 | 設定 | タスク |
| --- | --- | --- |
| index 0 | `r1pro_pick_up_radio.yaml` | `turning_on_radio` |
| index 1 | `r1pro_pick_up_trash.yaml` | `picking_up_trash` |
| index 2–49 | `b1k_<task_name>.yaml` | 各BEHAVIORタスク |

タスク名は、index順に次のとおりです。

| 区分 | タスク名（index順） |
| --- | --- |
| B10 | `turning_on_radio`, `picking_up_trash`, `putting_away_Halloween_decorations`, `cleaning_up_plates_and_food`, `can_meat`, `setting_mousetraps`, `hiding_Easter_eggs`, `picking_up_toys`, `rearranging_kitchen_furniture`, `putting_up_Christmas_decorations_inside` |
| B20 | `set_up_a_coffee_station_in_your_kitchen`, `putting_dishes_away_after_cleaning`, `preparing_lunch_box`, `loading_the_car`, `carrying_in_groceries`, `bringing_in_wood`, `moving_boxes_to_storage`, `bringing_water`, `tidying_bedroom`, `outfit_a_basic_toolbox` |
| B30 | `sorting_vegetables`, `collecting_childrens_toys`, `putting_shoes_on_rack`, `boxing_books_up_for_storage`, `storing_food`, `clearing_food_from_table_into_fridge`, `assembling_gift_baskets`, `sorting_household_items`, `getting_organized_for_work`, `clean_up_your_desk` |
| B40 | `setting_the_fire`, `clean_boxing_gloves`, `wash_a_baseball_cap`, `wash_dog_toys`, `hanging_pictures`, `attach_a_camera_to_a_tripod`, `clean_a_patio`, `clean_a_trumpet`, `spraying_for_bugs`, `spraying_fruit_trees` |
| B50 | `make_microwave_popcorn`, `cook_cabbage`, `chop_an_onion`, `slicing_vegetables`, `chopping_wood`, `cook_hot_dogs`, `cook_bacon`, `freeze_pies`, `canning_food`, `make_pizza` |

一般タスクの既定試行回数は20回、trashタスクは25回です。確認時だけ`--total-trials 1`のようにCLIで上書きできます。
