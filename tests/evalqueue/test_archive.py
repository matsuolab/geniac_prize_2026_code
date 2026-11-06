from __future__ import annotations

import tarfile

from capx.evalqueue.archive import build_archive
from capx.evalqueue.store import EvalStore


def test_the_archive_keeps_records_and_logs_but_not_scratch(tmp_path):
    store = EvalStore(tmp_path / "root")
    store.ensure_layout()
    (store.results / "r1.json").write_text("{}")
    (store.done / "j1.json").write_text("{}")
    run = store.runs / "r1" / "rs"
    for name in ("submission-home", "submission-out", "tmp-gateway"):
        (run / name).mkdir(parents=True)
        (run / name / "big.bin").write_text("x")
    (run / "gateway-results.json").write_text("{}")
    (run / "submission.log").write_text("log")

    target = build_archive(store, tmp_path / "out")

    with tarfile.open(target) as archive:
        names = archive.getnames()
    assert "results/r1.json" in names and "queue/done/j1.json" in names
    assert "runs/r1/rs/gateway-results.json" in names and "runs/r1/rs/submission.log" in names
    assert not any("big.bin" in name for name in names)
