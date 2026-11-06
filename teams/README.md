# チームの登録

予選のチームは運営が登録します。チームが pull request を送る必要はありません。
運営は、チームの HuggingFace の org に `capx-submission` を作り、`prelim/<org>.yaml` を足します。

```yaml
# prelim/team-alpha.yaml
team_id: team-alpha
repo: team-alpha/capx-submission
```

`team_id` は org の名前で固定です（[PRELIM_RULEBOOK.md](../PRELIM_RULEBOOK.md) §6.1）。
