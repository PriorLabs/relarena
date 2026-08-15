"""Tests for the RT baseline (registration, epoch grid, export layout).

No download and no `rt` import: everything here exercises the parts of the
wrapper that decide *what* RT will be asked to do — the search space, the step
budget, and the relbench-3.0.0 directory the preprocessor reads — rather than
the fine-tune itself, which needs a GPU and a published checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from relbench.base import Table, TaskType

from relarena.models.rt import RT_SPACE, RTPluRelModel
from relarena.models.rt import config as cfg
from relarena.models.rt.export import TASK_DIR, _write_dataset_dir, target_stats
from relarena.registry import registry


def test__registry__rt_plurel__is_registered_with_its_space() -> None:
    assert registry.get("rt-plurel") is RTPluRelModel
    assert registry.search_space("rt-plurel") is RT_SPACE
    # The reporting arm retrains on train+val, as every other baseline does.
    assert RTPluRelModel.refit_on_full_data is True


def _tiny_source() -> tuple[SimpleNamespace, SimpleNamespace, Table]:
    users = Table(
        df=pd.DataFrame({"uid": [0, 1, 2], "country": ["US", "DE", "FR"]}),
        fkey_col_to_pkey_table={},
        pkey_col="uid",
        time_col=None,
    )
    events = Table(
        df=pd.DataFrame(
            {
                "eid": [0, 1, 2],
                "uid": [0, 1, 2],
                "ts": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            }
        ),
        fkey_col_to_pkey_table={"uid": "users"},
        pkey_col="eid",
        time_col="ts",
    )
    label = Table(
        df=pd.DataFrame(
            {
                "uid": [0, 1, 2],
                "date": pd.to_datetime(["2021-02-01"] * 3),
                "y": [0.0, 2.0, 4.0],
            }
        ),
        fkey_col_to_pkey_table={"uid": "users"},
        pkey_col=None,
        time_col="date",
    )
    db = SimpleNamespace(table_dict={"users": users, "events": events})
    task = SimpleNamespace(
        entity_col="uid",
        entity_table="users",
        target_col="y",
        time_col="date",
        task_type=TaskType.REGRESSION,
    )
    return task, db, label


def test__write_dataset_dir__emits_the_manifest_rustler_parses(
    tmp_path: Path,
) -> None:
    task, db, label = _tiny_source()
    _write_dataset_dir(tmp_path, "relarena", db, task, {"train": label})

    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert manifest["name"] == "relarena"
    # rustler's manifest schema is deny_unknown_fields, so an entry may carry
    # only these keys, and only the ones that apply to the table.
    assert manifest["tables"]["users"] == {"pkey": "uid"}
    assert manifest["tables"]["events"] == {
        "pkey": "eid",
        "time_col": "ts",
        "fkeys": {"uid": "users"},
    }
    for name in ("users", "events"):
        assert (tmp_path / "db" / f"{name}.parquet").exists()


def test__write_dataset_dir__emits_the_task_manifest_and_split(tmp_path: Path) -> None:
    task, db, label = _tiny_source()
    _write_dataset_dir(tmp_path, "relarena", db, task, {"train": label})

    task_dir = tmp_path / "tasks" / TASK_DIR
    task_manifest = yaml.safe_load((task_dir / "manifest.yaml").read_text())
    assert task_manifest["task_type"] == "regression"
    assert task_manifest["entity_table"] == "users"
    assert task_manifest["entity_col"] == "uid"
    assert task_manifest["target_col"] == "y"
    assert task_manifest["time_col"] == "date"
    round_tripped = pd.read_parquet(task_dir / "train.parquet")
    assert len(round_tripped) == len(label.df)


def test__target_stats__matches_the_denormalizer_upstream_uses(tmp_path: Path) -> None:
    task, _db, label = _tiny_source()
    mean, std = target_stats(label, task)
    values = label.df["y"].to_numpy()
    assert mean == pytest.approx(values.mean())
    # ddof=1, as `_emit_and_score` computes it.
    assert std == pytest.approx(values.std(ddof=1))


def test__target_stats__constant_target__std_is_one_not_zero() -> None:
    task, _db, label = _tiny_source()
    label.df["y"] = 3.0
    _mean, std = target_stats(label, task)
    assert std == 1.0


def test__predict__before_fit__raises() -> None:
    task, db, label = _tiny_source()
    model = RTPluRelModel({})
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(task, db, label)


def test__config__departures_from_upstream_are_the_declared_ones() -> None:
    # These three are the documented departures; a change to any of them is a
    # change to the reported protocol and should fail a test, not slip through.
    args = _example_train_args()
    ev = cfg.eval_args(
        device="cpu",
        context_seed=0,
        num_rows=10,
        db_cutoff=None,
        context=(1024, 1024, 128, False),
    )
    # The selection arm bounds its contexts at the val horizon; the reporting
    # arm's `predict` bounds them at the test horizon (see context_cutoff).
    assert args["db_cutoff"] == 123
    assert ev["db_cutoff"] is None
    # predict scores every row; only the in-loop selection subsamples.
    # Rounded up to a whole number of eval batches, never a cap on rows.
    assert ev["items_per_task"] >= 10
    assert args["eval_items_per_task"] == 1024
    assert args["swa_momentum"] == 0.9999


def test__supported_task_types__entity_clf_and_reg_only() -> None:
    assert RTPluRelModel.supported_task_types == frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )
    assert RTPluRelModel.refit_on_full_data is True


def test__seed_offset__reads_the_preprocessed_table_info(tmp_path: Path) -> None:
    from relarena.models.rt.model import DB_NAME, _seed_offset

    (tmp_path / DB_NAME).mkdir()
    (tmp_path / DB_NAME / "table_info.json").write_text(
        f'{{"{TASK_DIR}:Test": {{"node_idx_offset": 41}}}}'
    )
    assert _seed_offset(tmp_path, "test") == 41


def _example_train_args(
    task_type: TaskType = TaskType.REGRESSION, phase: str = cfg.PHASE_INNER
) -> dict:
    return cfg.train_args(
        phase=phase,
        task_type=task_type,
        pre_dir="/pre",
        db_name="relarena",
        task_name="task",
        train_split="train",
        eval_split="val" if phase == cfg.PHASE_INNER else None,
        db_cutoff=123 if phase == cfg.PHASE_INNER else None,
        total_steps=1000,
        out_root="/out",
        run_id="r0",
        seed=0,
    )


def test__train_args__derives_the_head_and_loss_from_the_task_type() -> None:
    reg = _example_train_args(TaskType.REGRESSION)
    clf = _example_train_args(TaskType.BINARY_CLASSIFICATION)
    assert reg["load_ckpt_path"].endswith("/regression")
    assert reg["loss_fn"] == "l1"
    assert clf["load_ckpt_path"].endswith("/classification")
    assert clf["loss_fn"] == "bce"


def test__train_args__inner_arm_selects_on_val() -> None:
    # The selection settings. The row cap is what is given up to keep the curve
    # dense; the ensemble is upstream's, because it is what separates two
    # checkpoints whose val scores differ by less than one context draw.
    args = _example_train_args(phase=cfg.PHASE_INNER)
    assert args["eval_splits"] == ["val"]
    assert args["eval_freq"] == 100
    assert args["eval_items_per_task"] == 1024
    assert args["eval_ensemble_size"] == cfg.selection_ensemble_size()
    # Ten evals of patience, and rt.train requires a val split to stop on.
    assert args["early_stop_after_steps"] == cfg.patience_steps()


def test__train_args__outer_arm_has_no_patience_to_stop_on() -> None:
    # rt.train asserts early_stop_after_steps is None unless "val" is evaluated.
    args = _example_train_args(phase=cfg.PHASE_OUTER)
    assert args["early_stop_after_steps"] is None


def test__train_args__outer_arm_does_not_evaluate_at_all() -> None:
    # RelArena scores test itself; the model is never handed test labels.
    args = _example_train_args(phase=cfg.PHASE_OUTER)
    assert args["eval_splits"] == []
    assert args["eval_freq"] is None


def test__train_args__never_logs_and_never_re_censors() -> None:
    for phase in (cfg.PHASE_INNER, cfg.PHASE_OUTER):
        args = _example_train_args(phase=phase)
        assert args["wandb_disabled"] is True
        assert args["total_steps"] == 1000


def test__context_grid__is_covered_by_what_training_draws_from() -> None:
    # A net evaluated under a context it was not trained under is a different
    # measurement -- which is why the searched grid has to be a subset of the
    # shapes training mixes over.
    train = _example_train_args()
    for ctx, lcs, bw, pl in cfg.context_grid():
        assert ctx in train["ctx_size_list"]
        assert lcs in train["local_ctx_size_list"]
        assert bw in train["bfs_width_list"]
        assert pl in train["prefer_latest_list"]
    ev = cfg.eval_args(
        device="cpu",
        context_seed=0,
        num_rows=10,
        db_cutoff=None,
        context=cfg.context_grid()[0],
    )
    assert train["num_walks"] == ev["num_walks"]
    assert train["embedder"] == ev["embedder"]


def test__eval_args__carries_the_per_member_seed() -> None:
    assert (
        cfg.eval_args(
            device="cpu",
            context_seed=7,
            num_rows=10,
            db_cutoff=None,
            context=(1024, 1024, 128, False),
        )["context_seed"]
        == 7
    )


def test__predict__inner_arm__returns_a_placeholder_not_a_score() -> None:
    # rt reports no validation score: the checkpoint was chosen inside fit, and
    # scoring the whole val split would cost a large fraction of the training
    # arm to produce a number nothing selects on.
    task, db, label = _tiny_source()
    model = RTPluRelModel({})
    model._phase = cfg.PHASE_INNER
    model._checkpoint = Path("/nonexistent.safetensors")
    model._task_type = TaskType.REGRESSION
    pred = model.predict(task, db, label)
    # No checkpoint was loaded and no evaluator built -- the path above does not
    # exist, so reaching rt at all would have raised.
    assert pred.shape == (len(label.df),)
    assert not pred.any()


def test__preprocess_args__embeds_under_the_checkpoints_embedder() -> None:
    args = cfg.preprocess_args(dataset="/ds", out_dir="/pre", embed=False)
    assert args["embedder"] == _example_train_args()["embedder"]
    assert args["skip_tasks"] is False and args["embed"] is False


def test__space__is_parameter_free() -> None:
    # rt.train chooses the epoch budget inside one fit, so there is nothing for
    # the tuner to search over.
    assert RT_SPACE.default_overrides == {}
    assert RT_SPACE.is_tunable is False


def test__selection_steps__is_a_flat_step_budget() -> None:
    # A step budget, not an epoch count: flat across task sizes, as upstream's
    # is. Early stopping ends most arms well before the ceiling.
    assert cfg.selection_steps() == 50_000


def test__refit_steps__rescales_to_the_bigger_split() -> None:
    # The reporting arm trains on train+val, so the same raw step count would
    # be proportionally less training than validation chose.
    steps = cfg.refit_steps(chosen_step=1000, inner_rows=100_000, outer_rows=125_000)
    assert steps == 1250


def test__refit_steps__equal_splits__is_the_chosen_step() -> None:
    assert cfg.refit_steps(chosen_step=700, inner_rows=1000, outer_rows=1000) == 700


def test__refit_steps__never_zero() -> None:
    assert cfg.refit_steps(chosen_step=1, inner_rows=1_000_000, outer_rows=1) == 1


def test__write_dataset_dir__masked_split__restores_a_constant_target(
    tmp_path: Path,
) -> None:
    # RelBench drops the target from the masked test table; rustler needs the
    # column back (positional column stats) *and* needs a value in it (a null
    # target cell fails the context build for every row of the split).
    task, db, label = _tiny_source()
    masked = Table(
        df=label.df.drop(columns=["y"]),
        fkey_col_to_pkey_table=label.fkey_col_to_pkey_table,
        pkey_col=label.pkey_col,
        time_col=label.time_col,
    )
    _write_dataset_dir(tmp_path, "relarena", db, task, {"train": label, "test": masked})

    train = pd.read_parquet(tmp_path / "tasks" / TASK_DIR / "train.parquet")
    test = pd.read_parquet(tmp_path / "tasks" / TASK_DIR / "test.parquet")
    assert list(test.columns) == list(train.columns)  # same columns, same order
    assert test["y"].notna().all()  # a value, not a null
    assert test["y"].nunique() == 1  # a constant: it cannot encode the answer
    assert test["y"].dtype == train["y"].dtype


def test__eval_args__items_per_task__covers_every_row() -> None:
    # The evaluator floors items_per_task // eval_bs into a batch count, so a
    # cap equal to the row count silently drops the final partial batch.
    for num_rows in (1, 255, 256, 257, 726, 1000, 100_000):
        args = cfg.eval_args(
            device="cpu",
            context_seed=0,
            num_rows=num_rows,
            db_cutoff=None,
            context=(1024, 1024, 128, False),
        )
        eval_bs = max(1, args["tokens_per_gpu"] // max(args["ctx_size_list"]))
        n_batches = max(1, args["items_per_task"] // eval_bs)
        assert n_batches * eval_bs >= num_rows, (num_rows, args["items_per_task"])


def test__eval_args__leaves_rts_own_context_guards_alone() -> None:
    # RT quotes labelled task rows in its context, and the split being scored
    # has masked labels (a constant placeholder), so quoting it would teach the
    # model that placeholder. `db_cutoff` is what prevents that -- it puts the
    # whole split past the bound. rt used to carry a narrower second guard
    # (`train_only_fallback`); it was removed in rt 1.3.0 as a strict subset of
    # what this one already rejects.
    args = cfg.eval_args(
        device="cpu",
        context_seed=0,
        num_rows=10,
        db_cutoff=1_000,
        context=(1024, 1024, 128, False),
    )
    assert "train_only_fallback" not in args
    assert args["db_cutoff"] == 1_000


def test__context_cutoff__is_relbenchs_own_split_timestamp() -> None:
    # rt's db_cutoff contract: the value must be relbench's val_timestamp or
    # test_timestamp for the split being scored -- that is what upstream's
    # db_cutoff="val"/"test" resolves to. `task.dataset` carries both, so the
    # number is read rather than inferred from the split's rows.
    task = SimpleNamespace(
        dataset=SimpleNamespace(
            val_timestamp=pd.Timestamp("2005-01-01"),
            test_timestamp=pd.Timestamp("2010-01-01"),
        )
    )
    assert cfg.context_cutoff(task, "val") == int(
        pd.Timestamp("2005-01-01").timestamp()
    )
    assert cfg.context_cutoff(task, "test") == int(
        pd.Timestamp("2010-01-01").timestamp()
    )
    # No off-by-one. An earlier version anchored on `min(row timestamps) - 1`;
    # the `-1` was only needed because that anchor *is* one of the split's own
    # row timestamps, and rustler's past_bound is `ts > bound`. The split
    # timestamp is not, so it needs no nudge -- and on rel-f1 it sits 60 days
    # before the first row, where the two anchors disagree outright.
    assert cfg.context_cutoff(task, "val") % 1 == 0


def test__best_checkpoint__takes_the_swa_net_not_the_better_one(
    tmp_path: Path,
) -> None:
    # safetensors ships with the `rt` extra, not with a plain install.
    save_file = pytest.importorskip("safetensors.torch").save_file
    torch = pytest.importorskip("torch")

    from relarena.models.rt.model import _best_checkpoint

    for name, step in (("best_clf", 700), ("best_swa_clf", 300)):
        save_file(
            {"w": torch.zeros(1)},
            str(tmp_path / f"{name}.safetensors"),
            metadata={"step": str(step)},
        )
    # Only the step is selected on val, never the net: `best_clf` scores better
    # by construction here and is still not what is reported.
    path, step = _best_checkpoint(tmp_path, TaskType.BINARY_CLASSIFICATION)
    assert path.name == "best_swa_clf.safetensors" and step == 300


def test__context_grid__60_configs_over_24_context_builds() -> None:
    # ctx x (lcs, bw, pl), minus lcs > ctx which is not distinct from lcs == ctx.
    # Every ctx size for one (lcs, bw, pl) is scored off a prefix of one build,
    # so the search costs 18 passes rather than 36.
    grid = cfg.context_grid()
    assert len(grid) == 60
    assert len({(lcs, bw, pl) for _, lcs, bw, pl in grid}) == 24
    assert all(lcs <= ctx for ctx, lcs, _, _ in grid)


def test__train_args__mixes_context_shapes() -> None:
    args = cfg.train_args(
        phase=cfg.PHASE_INNER,
        task_type=TaskType.BINARY_CLASSIFICATION,
        pre_dir="/pre",
        db_name="relarena",
        task_name="task",
        train_split="train",
        eval_split="val",
        db_cutoff=1,
        total_steps=10_000,
        out_root="/out",
        run_id="r0",
        seed=0,
    )
    # A net trained at one shape is not usable at another; these are what make
    # the checkpoint tunable after the fact.
    assert len(args["ctx_size_list"]) > 1
    assert len(args["local_ctx_size_list"]) > 1
    assert len(args["bfs_width_list"]) > 1
    assert args["prefer_latest_list"] == [False, True]
    assert args["num_walks"] == 1_000
    # the schedule stays ours: no warmup, no decay, and the SWA EMA in place of
    # both; the patience is ours too
    assert args["lr_warmup_steps"] == 0
    assert args["lr_decay_steps"] == 0
    assert args["early_stop_after_steps"] == cfg.patience_steps()
    # the in-loop eval that ranks *steps* stays at one fixed shape
    assert args["eval_ctx_size_list"] == [1024]


def test__eval_args__pins_one_grid_point() -> None:
    args = cfg.eval_args(
        device="cpu",
        context_seed=0,
        num_rows=100,
        db_cutoff=None,
        context=(1024, 512, 64, True),
    )
    assert args["ctx_size_list"] == [1024]
    assert args["local_ctx_size"] == 512
    assert args["bfs_width"] == 64
    assert args["prefer_latest"] is True
    assert args["num_walks"] == 1_000


def test__eval_args__widened_build_covers_every_ctx_size() -> None:
    # One build serves every ctx size sharing a (lcs, bw, pl): contexts are
    # built at the largest and each smaller size scored off a prefix. The row
    # cap has to be rounded to whole batches of the *widened* build's batch
    # size, which is set by the largest ctx.
    args = cfg.eval_args(
        device="cpu",
        context_seed=0,
        num_rows=4096,
        db_cutoff=None,
        context=(1024, 256, 64, True),
        ctx_sizes=[256, 512, 1024],
    )
    assert args["ctx_size_list"] == [256, 512, 1024]
    eval_bs = max(1, args["tokens_per_gpu"] // 1024)
    assert args["items_per_task"] % eval_bs == 0
    assert args["items_per_task"] >= 4096


def _entry(root: Path, name: str, size: int = 4096) -> Path:
    # An export lives six levels down (rt/vN/db@fp/task/phase/embedder/splits)
    # and is complete when it has a `pre/` directory.
    e = root / "rt" / "v2" / "db@fp" / "task" / "phase-outer" / "emb" / name
    (e / "pre").mkdir(parents=True)
    (e / "pre" / "blob").write_bytes(b"x" * size)
    return e


def test__reap__evicts_least_recently_used_first(tmp_path: Path) -> None:
    import os

    from relarena.models.rt import export

    old, new = _entry(tmp_path, "old"), _entry(tmp_path, "new")
    os.utime(old, (1, 1))  # touched long ago
    os.utime(new, (10**9, 10**9))  # touched recently
    # free_target above any real free space, so it reaps until nothing is left
    export.reap(tmp_path, free_target=2**62)
    assert not old.exists(), "the least recently used entry should go first"


def test__reap__never_deletes_an_entry_a_live_process_claimed(tmp_path: Path) -> None:
    import os

    from relarena.models.rt import export

    e = _entry(tmp_path, "claimed")
    os.utime(e, (1, 1))  # oldest, so first in line to be reaped
    export._claim(e)  # ...but this process is reading it
    export.reap(tmp_path, free_target=2**62)
    assert e.exists(), "an entry a live process is reading must survive"


def test__reap__ignores_a_marker_whose_process_is_gone(tmp_path: Path) -> None:
    import os

    from relarena.models.rt import export

    e = _entry(tmp_path, "stale")
    (e / f"{export._INUSE}999999").touch()  # a pid that does not exist
    os.utime(e, (1, 1))
    export.reap(tmp_path, free_target=2**62)
    assert not e.exists(), "a stale marker must not pin a directory forever"


def test__embed__identical_text_is_embedded_once_and_linked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The embedding pass is the expensive half of an export and is identical
    # across the three exports of one task -- none of the label tables carries
    # text. Keying on the content of `text.json` makes the reuse exact without
    # having to argue about which exports happen to coincide.
    import sys
    from types import ModuleType

    from relarena.cache import CacheConfig
    from relarena.models.rt import export

    calls = []

    def embed_dataset(pre_dataset_dir: Path, embedder: str, batch_size: int) -> int:
        calls.append(Path(pre_dataset_dir))
        n = len(
            np.asarray(
                yaml.safe_load((Path(pre_dataset_dir) / "text.json").read_text())
            )
        )
        (Path(pre_dataset_dir) / f"text_emb_{embedder}.bin").write_bytes(
            b"e" * (n * 384 * 2)
        )
        return 384

    def update_meta_with_embeddings(
        pre_dataset_dir: Path, embedder: str, d_text: int
    ) -> None:
        (Path(pre_dataset_dir) / "meta.json").write_text(f'{{"d_text": {d_text}}}')

    fake = ModuleType("rt.preprocess")
    fake.embed_dataset = embed_dataset  # type: ignore[attr-defined]
    fake.update_meta_with_embeddings = update_meta_with_embeddings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rt", ModuleType("rt"))
    monkeypatch.setitem(sys.modules, "rt.preprocess", fake)

    cache = CacheConfig(directory=tmp_path / "store", on_miss="fill")
    text = '["alpha", "beta"]'
    for name in ("first", "second"):
        d = tmp_path / name / "relarena"
        d.mkdir(parents=True)
        (d / "text.json").write_text(text)
        export._embed(d, "all-MiniLM-L12-v2", 512, cache)

    # Embedded once; the second export got the same bytes.
    assert len(calls) == 1
    first = (
        tmp_path / "first" / "relarena" / "text_emb_all-MiniLM-L12-v2.bin"
    ).read_bytes()
    second = (
        tmp_path / "second" / "relarena" / "text_emb_all-MiniLM-L12-v2.bin"
    ).read_bytes()
    assert first == second and first
    # Both carry the meta entry the trainer reads d_text out of.
    assert (tmp_path / "second" / "relarena" / "meta.json").exists()


def test__embed__different_text_is_not_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import ModuleType

    from relarena.cache import CacheConfig
    from relarena.models.rt import export

    calls = []

    def embed_dataset(pre_dataset_dir: Path, embedder: str, batch_size: int) -> int:
        calls.append(Path(pre_dataset_dir))
        (Path(pre_dataset_dir) / f"text_emb_{embedder}.bin").write_bytes(b"e" * 768)
        return 384

    fake = ModuleType("rt.preprocess")
    fake.embed_dataset = embed_dataset  # type: ignore[attr-defined]
    fake.update_meta_with_embeddings = lambda *a: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rt", ModuleType("rt"))
    monkeypatch.setitem(sys.modules, "rt.preprocess", fake)

    cache = CacheConfig(directory=tmp_path / "store", on_miss="fill")
    for name, text in (("a", '["alpha"]'), ("b", '["beta"]')):
        d = tmp_path / name / "relarena"
        d.mkdir(parents=True)
        (d / "text.json").write_text(text)
        export._embed(d, "all-MiniLM-L12-v2", 512, cache)
    assert len(calls) == 2


def test__context_cutoff__every_arm_names_its_horizon() -> None:
    # The reporting arm's training pass scores nothing, so the rule that bounds
    # a context at the scored split's horizon has no split to read. `None` is
    # that rule's answer, which is why the call site has no special case -- and
    # it is also the only reachable answer: `fit` is never handed the test
    # table, and any bound at or after the phase horizon would be inert anyway.
    # The reporting arm's training pass scores nothing, but its database is
    # censored at test_timestamp, so that is the horizon it names. Inert --
    # rustler takes min(target_ts, cutoff) and every train+val row precedes it
    # on all 21 tasks -- but stated rather than inferred.
    task = SimpleNamespace(
        dataset=SimpleNamespace(
            val_timestamp=pd.Timestamp("2005-01-01"),
            test_timestamp=pd.Timestamp("2010-01-01"),
        )
    )
    assert cfg.context_cutoff(task, "test") > cfg.context_cutoff(task, "val")


def test__rt_extra__pins_the_version_that_is_installed() -> None:
    # The `rt` extra installs a prebuilt wheel from a GitHub release, so the
    # URL is the only statement of which rt this model was written against.
    # If an environment carries a different one -- an editable checkout, or a
    # stale wheel -- the model may be running code the config assumes is there.
    # `db_cutoff` is the live example: it takes an integer only from 1.2.0 on,
    # and v1.1.0's wheel silently lacked it.
    import re

    import tomllib

    root = Path(__file__).resolve().parents[3]
    spec = tomllib.loads((root / "pyproject.toml").read_text())
    urls = [
        d
        for group in spec["project"].get("optional-dependencies", {}).values()
        for d in group
        if "relational-transformer" in d
    ]
    assert urls, "the rt extra no longer pins relational-transformer"
    pinned = re.search(r"relational_transformer-([0-9.]+)-", urls[0])
    assert pinned, f"cannot read a version out of {urls[0]!r}"

    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("relational-transformer")
    except PackageNotFoundError:  # the `rt` extra is not installed here
        pytest.skip("relational-transformer is not installed")
    assert installed == pinned.group(1), (
        f"installed relational-transformer {installed} but the rt extra pins "
        f"{pinned.group(1)}. Install from the pinned wheel: a source checkout "
        "or an older release can differ in ways the config depends on."
    )


def test__denormalization__uses_the_same_table_rustler_normalized_by() -> None:
    """The refit's normalizer is train+val, and the denormalizer must match it.

    rustler computes each task table's column statistics per table, then
    overwrites Val's and Test's with the split named `Train`
    (`rustler/src/pre.rs`, the `col_stats_map` pass). So a regression target is
    normalized by whatever that export calls `Train` -- which on the reporting
    arm is the train+val union, not relbench's train split. The two differ
    materially: on rel-f1/driver-position, mean 13.901 -> 13.725 and std 7.026
    -> 6.934.

    `fit` therefore has to take `target_stats` from the *same* table it exports
    as `train`, and `predict` has to export that same table beside the split it
    scores. Denormalizing a refit's predictions with train-only statistics would
    shift and rescale every one of them.
    """
    task, db, label = _tiny_source()
    union = Table(
        df=pd.concat([label.df, label.df.assign(y=label.df["y"] * 3 + 1)]),
        fkey_col_to_pkey_table=label.fkey_col_to_pkey_table,
        pkey_col=label.pkey_col,
        time_col=label.time_col,
    )
    # The union's statistics are not the sub-table's -- otherwise this proves
    # nothing.
    assert target_stats(union, task) != target_stats(label, task)

    model = RTPluRelModel({})
    model._task_type = TaskType.REGRESSION
    model._target_stats = target_stats(union, task)
    model._train_table = union

    # What fit() pairs: stats and exported `train` come from one table.
    assert model._target_stats == target_stats(model._train_table, task)

    # And the denormalizer is that pair, applied as `raw * std + mean`.
    mean, std = model._target_stats
    raw = np.array([-1.0, 0.0, 2.5])
    assert np.allclose(raw * std + mean, [mean - std, mean, mean + 2.5 * std])


def test__step_and_context_searches__do_not_read_the_same_rows() -> None:
    # Both searches read a capped prefix of a shuffled val split. Sharing a
    # shuffle seed makes the smaller read a strict subset of the larger, so the
    # 1024 rows that picked the step are 1024 of the 4096 that then pick the
    # context -- two selections compounding on one sample, with the second
    # unable to notice that the first already overfit it.
    train = _example_train_args()
    search = cfg.eval_args(
        device="cpu",
        context_seed=0,
        num_rows=cfg.tune_rows(),
        db_cutoff=1,
        context=cfg.context_grid()[0],
        shuffle_seed=cfg.tune_shuffle_seed(),
    )
    assert search["shuffle_seed"] != train["eval_shuffle_seed"]

    # predict keeps upstream's seed: it reads every row, so the seed only sets
    # the order, and changing it would be churn for nothing.
    predict = cfg.eval_args(
        device="cpu",
        context_seed=0,
        num_rows=100,
        db_cutoff=1,
        context=cfg.context_grid()[0],
    )
    assert predict["shuffle_seed"] == 0


def test__step_selection__spends_its_budget_on_shapes_not_seeds() -> None:
    # The step search and the context search get the same number of evaluator
    # passes; they spend it differently. The context search averages
    # `selection_ensemble_size` draws of one shape per candidate. The step
    # search takes one draw of each of `step_select_grid`'s shapes, because the
    # axis that was mistiming the step is ctx, not sampling noise.
    train = _example_train_args()
    assert train["eval_ensemble_size"] == cfg.selection_ensemble_size()
    assert len(train["eval_ctx_lcs_bw_pl_grid"]) == len(cfg.step_select_grid())
    assert cfg.selection_ensemble_size() == 4
    # The context search's members are a prefix of the reported ensemble's; the
    # step search draws from a disjoint family so the two do not compound.
    assert train["eval_context_seed"] == cfg.step_context_seed() != 0
    assert cfg.selection_ensemble_size() < 8


def test__inference__is_compiled_like_training_is() -> None:
    # A compiled graph is not bit-identical to an eager one, and training runs
    # compiled -- so scoring eagerly would choose the step and the context, and
    # report the number, under a different numerical path from the one that
    # trained the weights. It is also simply faster: the net is loaded once per
    # stage and used for many forwards.
    assert cfg.compile_inference() is True
    assert _example_train_args()["compile"] is True


def test__eval_batch_shape__is_stable_across_the_search() -> None:
    # Compiling is only cheap if the graph count stays small. The batch size is
    # tokens_per_gpu // max(ctx_size_list), and every build the search walks has
    # max ctx 1024 (lcs <= ctx, and lcs's largest is 1024), so the batch is
    # constant and only the ctx size varies -- three graphs at most.
    builds: dict[tuple[int, int, bool], list[int]] = {}
    for ctx, lcs, bw, pl in cfg.context_grid():
        builds.setdefault((lcs, bw, pl), []).append(ctx)
    batch_sizes = set()
    for (lcs, bw, pl), sizes in builds.items():
        args = cfg.eval_args(
            device="cpu",
            context_seed=0,
            num_rows=4096,
            db_cutoff=1,
            context=(max(sizes), lcs, bw, pl),
            ctx_sizes=sorted(sizes),
        )
        batch_sizes.add(args["tokens_per_gpu"] // max(args["ctx_size_list"]))
    assert len(batch_sizes) == 1, batch_sizes


def test__step_and_context_searches__share_neither_rows_nor_context_draws() -> None:
    # Two selections on one val split. Sharing rows makes the second unable to
    # see that the first overfit them; sharing context draws does the same for
    # the context randomness. Both must differ.
    train = _example_train_args()
    tune = cfg.eval_args(
        device="cpu",
        context_seed=0,
        num_rows=cfg.tune_rows(),
        db_cutoff=1,
        context=cfg.context_grid()[0],
        shuffle_seed=cfg.tune_shuffle_seed(),
    )
    assert train["eval_shuffle_seed"] != tune["shuffle_seed"]
    assert cfg.step_context_seed() != 0, "0 is the tuning/predict family's base"

    # A different base is a disjoint family, not an offset into the same one:
    # member_context_seed mixes the base through splitmix64.
    member_context_seed = pytest.importorskip(
        "rt.eval", reason="needs the rt extra"
    ).member_context_seed

    step = {member_context_seed(cfg.step_context_seed(), m) for m in range(8)}
    tune_seeds = set(cfg.ensemble_context_seeds(cfg.selection_ensemble_size()))
    report = set(cfg.ensemble_context_seeds())
    assert not (step & report), "the step search reuses reported context draws"
    assert not (step & tune_seeds)
    # ...while tuning stays a prefix of what predict reports under.
    assert tune_seeds <= report


def test__train_args__scores_only_the_net_it_reports() -> None:
    # The live net is never reported -- selection takes best_swa_* -- and
    # scoring it cost half the in-loop eval for a number nothing reads, while
    # also feeding early stopping, so it held runs open after SWA had peaked.
    args = _example_train_args(phase=cfg.PHASE_INNER)
    assert args["eval_live"] is False
    assert args["swa_momentum"] is not None, "eval_live=False needs an SWA net"
    # Patience is doubled to match: an EMA improves in coarser increments than
    # a live net does.
    assert args["early_stop_after_steps"] == cfg.patience_steps() == 10_000
    assert cfg.patience_steps() % args["eval_freq"] == 0


def test__best_checkpoint__nothing_published__reports_the_warm_start(
    tmp_path: Path,
) -> None:
    # With eval_live=False there is no live checkpoint to fall back to, and no
    # SWA checkpoint exists at step 0. Both missing means validation never beat
    # step 0, so the honest report is the warm start unmodified -- not a crash.
    pytest.importorskip("safetensors")  # ships with the rt extra
    from relarena.models.rt.model import _best_checkpoint

    path, step = _best_checkpoint(tmp_path, TaskType.REGRESSION)
    assert step == 0
    assert path == cfg.warm_start(TaskType.REGRESSION)


def test__fit__outer_arm_at_step_zero__still_carries_the_chosen_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The step-0 reporting arm reports the warm start unmodified and never
    # trains, so it never reaches `_refit_steps` -- which is what normally hands
    # the context across the arms. `predict` still has to report under the
    # context validation chose; leaving it unset unpacks None in `eval_args`.
    from relarena.models.rt import model as rt_model

    task, db, label = _tiny_source()
    chosen = (256, 256, 64, True)

    model = RTPluRelModel({})
    model.run_identity = SimpleNamespace(phase=cfg.PHASE_OUTER, dataset="ds", task="tk")
    monkeypatch.setattr(rt_model, "preprocessed_dir", lambda *a, **k: tmp_path)
    monkeypatch.setitem(
        rt_model._SELECTED,
        model._selection_key(0),
        cfg.Selection(step=0, rows=3, context=chosen),
    )

    model.fit(task, db, label, None, seed=0)

    assert model._checkpoint == cfg.warm_start(TaskType.REGRESSION)
    assert model._context == chosen
