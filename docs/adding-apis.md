# モデルが利用する新しいAPIの追加

[公式ドキュメント](https://github.com/capgym/cap-x/blob/main/docs/adding-apis.md)

APIは、LLMが生成するPythonコードから呼び出せる関数群です。

ロボットの移動、グリッパ開閉、観測取得、物体検出、把持計画など、モデルがタスク実行時に利用できる能力を定義します。

## Step 1: APIを実装する

`ApiBase`を継承し、モデルへ公開する関数を`functions()`で指定します。

例えば、`capx/integrations/my_robot/control.py`に以下のようなAPIを実装します。

```python
# capx/integrations/my_robot/control.py

from typing import Any

import numpy as np

from capx.integrations.base_api import ApiBase


class MyRobotControlApi(ApiBase):
    def __init__(self, env):
        super().__init__(env)

        # 必要に応じて、知覚モデルやIKクライアントを初期化する
        # self.ik_client = ...
        # self.detector = ...

    def functions(self) -> dict[str, Any]:
        """LLM生成コードへ公開する関数を返す。"""
        return {
            "get_observation": self.get_observation,
            "move_to": self.move_to,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_observation(self) -> dict[str, Any]:
        """現在のカメラ画像とロボット状態を取得する。"""
        return self._env.get_observation()

    def move_to(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> None:
        """エンドエフェクタを目標姿勢へ移動する。

        Args:
            position:
                ワールド座標系の目標位置。
                Shape: (3,), 単位: m。
            orientation:
                ワールド座標系の目標姿勢。
                Shape: (4,), 順序: [w, x, y, z]。
        """
        position = np.asarray(
            position,
            dtype=np.float64,
        ).reshape(3)

        orientation = np.asarray(
            orientation,
            dtype=np.float64,
        ).reshape(4)

        # 低レベル環境の制御関数を呼び出す
        self._env.move_to_pose_blocking(
            position,
            orientation,
        )

    def open_gripper(self) -> None:
        """グリッパを開く。"""
        self._env.open_gripper_blocking()

    def close_gripper(self) -> None:
        """グリッパを閉じる。"""
        self._env.close_gripper_blocking()
```

`self._env`には、API生成時に渡された低レベル環境が格納されています。

## 既存例: `FrankaControlApiReduced`

既存の`FrankaControlApiReduced`では、公開関数をおおよそ以下のように定義しています。

```python
class FrankaControlApiReduced(ApiBase):
    def functions(self):
        return {
            "get_observation": self.get_observation,
            "segment_sam3_text_prompt":
                self.segment_sam3_text_prompt,
            "plan_grasp": self.plan_grasp,
            "solve_ik": self.solve_ik,
            "move_to_joints": self.move_to_joints,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }
```

LLMは、公開された関数を組み合わせて、例えば以下のようなコードを生成します。

```python
obs = get_observation()

masks = segment_sam3_text_prompt(
    obs["robot0_robotview"]["images"]["rgb"],
    text_prompt="red cube",
)

grasp_poses, scores = plan_grasp(
    depth=obs["robot0_robotview"]["images"]["depth"],
    intrinsics=obs["robot0_robotview"]["intrinsics"],
    segmentation=masks[0]["mask"],
)

joints = solve_ik(
    target_position,
    target_quaternion,
)

move_to_joints(joints)
close_gripper()
```

APIは、タスクの手順全体ではなく、モデルが組み合わせて利用する操作や知覚の関数を定義します。

## Step 2: APIを登録する

作成したAPIを利用可能にするため、`capx/integrations/__init__.py`へ登録します。

```python
from .my_robot.control import MyRobotControlApi

register_api(
    "MyRobotControlApi",
    MyRobotControlApi,
)
```

## Step 3: タスク設定でAPIを指定する

APIを使用するタスクYAMLの`apis`へ、登録したAPI名を追加します。

```yaml
env:
  cfg:
    apis:
      - MyRobotControlApi
```

これにより、`functions()`で公開した関数がLLMへ提示され、生成コードから利用可能になります。

## Step 4: 動作を確認する

以下のコマンドで、登録済みAPIを確認できます。

```bash
uv run --no-sync --active python - <<'PY'
from capx.integrations import list_apis

print(list_apis())
PY
```

モデルへ提示される関数シグネチャとdocstringは、`combined_doc()`で確認できます。

```python
api = MyRobotControlApi(fake_env)

print(api.combined_doc())
```
