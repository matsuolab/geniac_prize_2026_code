# スキルライブラリのコンパイル

[公式ドキュメント](https://github.com/capgym/cap-x/blob/main/scripts/skill_library_compilation/README.md)

評価（eval）の出力を分析し，成功したトライアルから再利用可能なスキルライブラリをコンパイルします．

# 何をするか

使用するAPIが低レベルの時，LLMはより高レベルな認識・制御のための関数を自分で生成します．

ここでは過去の評価において生成された関数群のうち再利用可能なものをまとめ，スキルライブラリを作成します．

関数名や生成回数から再利用可能な関数を厳選し，LLMによりカテゴリ分け・整形・docstringの付与を行います．

作成したスキルライブラリを評価で利用したい場合は [Adding APIs](/docs/adding-apis.md)に従ってAPIを作成してください．


## 前提条件

- OpenRouter プロキシが起動していること: `uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey`（`model.sock` で待ち受けます）
- `outputs/` に以下の条件を満たす評価の出力があること（`capx/envs/launch.py` によるもの）
    - 使用したAPIが低レベルである（FrankaLiberoApiReducedなど）
    - タスクに成功している

## 使い方

```bash
# 1. 出力をパース → experiment ごとに analysis.txt, highlights.txt, functions.txt を生成
uv run --no-sync --active scripts/skill_library_compilation/parse_outputs.py \
    --cfg.output-dir outputs/

# 2. モデルとタスクを横断して要約
uv run --no-sync --active scripts/skill_library_compilation/summarize_analysis.py \
    --cfg.output-dir outputs/

# 3. 成功した reduced-API の experiment からスキルライブラリをコンパイル
uv run --no-sync --active scripts/skill_library_compilation/compile_skill_library.py \
    --cfg.output-dir outputs/
```

出力: `outputs/skill_library.txt` — 今後の評価のために厳選された再利用可能な関数群．

## ユーティリティ

- **`trial_folder_rename.py`** — トライアルフォルダの番号を標準化する（`--cfg.dry-run` でプレビュー）
- **`eval_dir_to_code.py`** — レビュー用にすべての `code.py` ファイルを1つのファイルに統合する

各スクリプトは `--help` で全オプションを確認できる．
