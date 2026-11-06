# タスクパック

タスクパックは、gateway が読み込むタスクの集まりです（`pack.yaml` と env のコード）。
同じ gateway がどのパックでも動き、公開パックと非公開パックの違いは中身だけです
（`capx/protocol/taskpack.py`）。

| パック | 中身 | 公開 |
|---|---|---|
| `taskpacks/original` | CaP-X と同じタスク（Robosuite 7、LIBERO 130、BEHAVIOR 50） | 公開 |

予選と本選のパックは主催者だけが持ち、配布しません。同じ仕組みで書かれ、同じ評価環境で動きます。

`pack.yaml` の `limits` が、ジョブ全体の時間を計算するための1試行あたりの時間（`trial_time_s`）と、
試行回数・step 上限の既定値を決めます。タスクごとに `n_trials` と `trial_steplimit` を上書きできます。
ジョブ全体の時間は `trial_time_s` × 試行の総数です。1試行ごとの時間の上限はありません。

```yaml
pack_id: my-pack
limits:
  trial_time_s: 300
  n_trials: 1
  trial_steplimit: 8000
register_modules:
  robosuite: [my_envs]              # <pack>/my_envs/ が register_env を呼ぶ
tasks:
  - task_id: cube_lifting__swap
    embodiment: franka
    simulator: robosuite
    instruction: Pick up the cube and lift it.
    env_target: my_envs_cube_lifting
    env_kwargs: {variation: swap}   # env のコンストラクタに渡す。提出物には送らない
    trial_steplimit: 1500
```

- `register_modules` を import するのは env を作るプロセスだけです。gateway はシミュレータを
  持たないので import しません。
- env が `max_steps` 引数を持つ場合、`trial_steplimit` に 10000 を足した値が渡されます。
  試行を終えるのは gateway が数える制御サイクルの数で、シミュレータの horizon ではありません
  （`reset` で物体を落ち着かせるために進めたステップも horizon に数えられるためです）。
  `env_kwargs` に `max_steps` は書けません。
- `pack.yaml` の未知のキーはエラーになります。
- `score_weights`（シミュレータ id → 正の数）を書くと、スコアはシミュレータごとの成功率の重み付き平均に
  なります（ルールブック §4.5）。書くならパックのすべてのシミュレータに付けます。重みは提出物にもリーダーボードにも
  出ません。書かなければ、すべての試行を同じ重さで数えます。
- 別のパックからタスクを選ぶだけのパックも書けます（`extends: <同じ階層のパック名>` と
  `task_ids`）。この場合も `limits` と `score_weights` は自分で書きます。
- 複数のパックで使う env のコードは、同じ階層の別ディレクトリに置いて `code_dirs` で参照します。
