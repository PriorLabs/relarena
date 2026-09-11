"""RelArena's censored database → RT's preprocessed tensor directory.

RT does not consume a `relbench.base.Database`. It consumes a directory of
rustler tensors plus text embeddings, produced by `rt.preprocess` from a
*relbench-3.0.0 dataset directory*::

    <dataset>/manifest.yaml            name + per-table pkey/time_col/fkeys
    <dataset>/db/<table>.parquet       one file per database table
    <dataset>/tasks/<task>/manifest.yaml
    <dataset>/tasks/<task>/{train,val,test}.parquet

This module writes that directory from the objects the harness hands a model,
then runs `rt.preprocess.one` over it.

**Why not the published `stanford-star/relbench-preprocessed`.** Because it is
preprocessed from *upstream's* database, and the harness's is not the same one:
RelArena drops fully-NaN columns, `Unnamed: N` row-index artifacts and
rel-ratebeer's time-leaking user aggregates
(`relarena.dataset.drop_noncanonical_columns`), and — the part that actually
matters — it censors the database at the **val** timestamp on the inner split.
Reusing a test-censored public artifact for the tuning phase would score
validation against a database that already contains the rows validation is
meant to predict. So the export is from the `db` the harness passed, whatever
it censored it to, and RT's own `db_cutoff` is left off (see `config.py`).

**Every export carries a train split.** rustler derives each task's column
statistics — the mean and standard deviation RT normalizes a target by — from
the `train` split alone, and copies them onto `val` and `test`
(`rustler/src/pre.rs`, the `col_stats_map` pass). An export holding only a
val or test split therefore has no statistics to inherit and aborts. That is a
constraint worth having: it means val and test targets are normalized by
*training* statistics, never their own, and it is why `target_stats` below
reproduces the denormalizer from the training table.

**The duplication, and what is shared.** The key below pins the task, because
the exported directory carries that task's label tables — so three directories
are built per task (the selection arm's, the refit's, and the test export beside
it), and a database with several tasks builds them several times over. Rustler
is rerun for each; that part is minutes.

The *embedding* pass is not minutes. It is a sentence-transformer over every
distinct string in the database — 19.5M of them on rel-amazon, ~8 minutes of
A100 per pass — and it is identical across all of those directories, because
none of the label tables carries text. `_embed` therefore keys embeddings on a
hash of the text list itself, in a store beside the directory cache: one pass
per distinct database-text, reused by hardlink everywhere else. Verified against
the built cache — the reporting arm's two exports and the selection arm's
exports of two tasks sharing a horizon all hash the same `text.json`.

**It filled a node's disk.** On 2026-08-12 three jobs on blackwell1 died with
`No space left on device` writing a checkpoint: its 438G disk was 99% full and
319G of that was this cache -- 162G of rel-amazon alone, four tasks x three
exports of the same database. `reap` below evicts unused entries when the disk
runs low, and the shared embedding store removes the largest single duplicate.

Collapsing the directories themselves would mean one export per (database,
phase) holding *every* task's splits -- rustler globs `tasks/*` and would ingest
them together. What stops it is the harness contract: a model is handed one
task, so no model can assemble the other tasks' label tables. It wants a warmer
that knows the whole task list, which `warm_cache.py` does; the model would then
read a directory it did not build.

**One artifact per (database, split set).** The expensive step is rustler plus
the text embedder, and it is keyed by the database *and* the label tables
exported beside it. `fit` exports what it trains on (plus the val split it
validates against); `predict` exports the table it scores beside the same train
split its model was fit on. Reaching into `task.get_table("test")` from `fit` to
save the second pass is deliberately not done: the harness chose what to hand
the model, and a model that goes around it is one audit away from a leak.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml
from relbench.base import Database, EntityTask, Table, TaskType

from relarena.cache import CacheConfig, CacheMiss, cache_key
from relarena.checksums import database_checksum, table_checksum
from relarena.identity import RunIdentity
from relarena.models.rt.config import preprocess_args

logger = logging.getLogger(__name__)

#: Bumped when what this module *writes* changes in a way that invalidates a
#: published tensor directory — the key pins the data, not the code shaping it.
EXPORT_VERSION = 2

#: relbench task types, as the task manifest spells them.
_TASK_TYPE: dict[TaskType, str] = {
    TaskType.BINARY_CLASSIFICATION: "binary_classification",
    TaskType.REGRESSION: "regression",
}

#: The task directory name inside every exported dataset. The task's real name
#: is in the cache key, not here: rustler reads the *directory* name as the
#: task name, and a fixed one keeps the RT-side task list a constant.
TASK_DIR = "task"

WARM_HINT = "Run `python -m relarena.models.rt.warm_cache` for this split first."


def _table_spec(table: Table) -> dict[str, object]:
    """One `tables:` entry of a dataset manifest."""
    spec: dict[str, object] = {}
    if table.pkey_col is not None:
        spec["pkey"] = table.pkey_col
    if table.time_col is not None:
        spec["time_col"] = table.time_col
    if table.fkey_col_to_pkey_table:
        spec["fkeys"] = dict(table.fkey_col_to_pkey_table)
    return spec


def _write_dataset_dir(
    directory: Path,
    name: str,
    db: Database,
    task: EntityTask,
    splits: dict[str, Table],
) -> None:
    """Write `db` + `splits` as a relbench-3.0.0 dataset directory.

    The manifest schema rustler parses is `deny_unknown_fields`, so this emits
    exactly the keys it names and nothing else — a stray key stops
    preprocessing rather than being ignored.
    """
    (directory / "db").mkdir(parents=True, exist_ok=True)
    tables: dict[str, dict[str, object]] = {}
    for table_name, table in db.table_dict.items():
        table.df.to_parquet(directory / "db" / f"{table_name}.parquet", index=False)
        tables[table_name] = _table_spec(table)
    (directory / "manifest.yaml").write_text(
        yaml.safe_dump({"name": name, "tables": tables}, sort_keys=True)
    )

    task_dir = directory / "tasks" / TASK_DIR
    task_dir.mkdir(parents=True, exist_ok=True)
    # The dtype the target is carried in, taken from a split that has it. Every
    # split has to agree on it: rustler copies the train split's per-column
    # statistics onto val and test *positionally*, so a column that changes type
    # or position between them is normalized by the wrong one's constants.
    target_dtype = next(
        (
            table.df[task.target_col].dtype
            for table in splits.values()
            if task.target_col in table.df.columns
        ),
        None,
    )
    for split_name, table in splits.items():
        df = table.df
        if task.target_col not in df.columns:
            # The masked test table has no target column at all — RelBench drops
            # it rather than nulling it, while RT's own preprocessed data keeps
            # the values. So the column is restored here, as a **constant**, not
            # as nulls: rustler builds a target cell per seed row and a null one
            # fails the build (every row of a split dropped with "build
            # failed/timed out"). The value is never read back — `predict`
            # ignores the labels the evaluator returns — and it cannot leak,
            # because it does not depend on the answer.
            filler = 0
            column = pd.Series(filler, index=df.index)
            if target_dtype is not None:
                column = column.astype(target_dtype)
            df = df.assign(**{task.target_col: column})
        df.to_parquet(task_dir / f"{split_name}.parquet", index=False)
    (task_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "name": TASK_DIR,
                # "forecast": the label is a future-looking quantity attached to
                # an entity at a timestamp, not a column of an existing table.
                "kind": "forecast",
                "task_type": _TASK_TYPE[task.task_type],
                "entity_table": task.entity_table,
                "entity_col": task.entity_col,
                "target_col": task.target_col,
                "time_col": task.time_col,
            },
            sort_keys=True,
        )
    )


def _key(
    db: Database,
    task: EntityTask,
    splits: dict[str, Table],
    identity: RunIdentity | None,
) -> PurePosixPath:
    """Content key over everything the tensor directory depends on."""
    dataset = "direct" if identity is None else identity.dataset
    db_fingerprint = (
        None if identity is None else identity.dataset_fingerprint
    ) or f"{int(database_checksum(db)):016x}"
    task_name = (
        f"{task.entity_table}-{task.target_col}"
        if identity is None or identity.task is None
        else identity.task
    )
    split_segments = [
        f"{name}-{int(table_checksum(table)):016x}"
        for name, table in sorted(splits.items())
    ]
    segments = [
        "rt",
        f"v{EXPORT_VERSION}",
        f"{dataset}@{db_fingerprint}",
        task_name,
        "phase-"
        + ("direct" if identity is None or identity.phase is None else identity.phase),
        f"embedder-{preprocess_args(dataset='', out_dir='', embed=False)['embedder']}",
        *split_segments,
    ]
    return cache_key(*segments)


#: Reap when the store's filesystem drops below this much free space. Chosen
#: against what one export of the largest RelBench database costs: rel-amazon's
#: is ~40G, so a job needs headroom of that order to publish one at all.
_REAP_FREE_BYTES = 80 * 1024**3

#: An entry in use is one a live process said it was reading. The marker is a
#: file named for that process; the reaper skips an entry while any of its
#: markers names a process that still exists, and ignores the rest -- a job
#: killed mid-read leaves a marker behind, and a stale marker must not pin a
#: directory forever.
_INUSE = ".inuse."


def _claim(entry: Path) -> None:
    """Mark `entry` as being read by this process."""
    try:
        (entry / f"{_INUSE}{os.getpid()}").touch()
        entry.touch()  # last-use, for the LRU order
    except OSError:
        pass  # a read-only or vanished store is not this function's problem


def _in_use(entry: Path) -> bool:
    """True while some live process has claimed `entry`."""
    for marker in entry.glob(f"{_INUSE}*"):
        try:
            pid = int(marker.name[len(_INUSE) :])
        except ValueError:
            continue
        try:
            os.kill(pid, 0)  # signal 0: existence probe, delivers nothing
        except ProcessLookupError:
            marker.unlink(missing_ok=True)  # its process is gone
            continue
        except PermissionError:
            pass  # alive, just another user's
        return True
    return False


def reap(root: Path, *, free_target: int = _REAP_FREE_BYTES) -> int:
    """Delete least-recently-used exports until `root` has `free_target` free.

    The store is node-local, fill-only and shared by every job on the node, and
    nothing else deletes from it: on 2026-08-12 that filled a 438G disk with
    319G of exports and killed three running jobs. This is the eviction that
    was missing.

    Least-recently-used by the entry's own mtime, which `_claim` touches when a
    job starts reading one. An entry a live process has claimed is never
    deleted, however old -- better to run out of space than to delete a
    directory out from under a job that is reading it. Returns bytes freed.
    """
    if not root.exists():
        return 0
    freed = 0
    # Found by their `pre/` marker, not by a fixed depth: a key carries one
    # segment per exported split, so an entry sits at depth 7 or 8 depending on
    # whether it holds one split or two. A depth-6 glob matched neither, which
    # is the sort of thing that makes a reaper quietly reap nothing.
    entries = [d.parent for d in root.rglob("pre") if d.is_dir()]
    for entry in sorted(entries, key=lambda d: d.stat().st_mtime):
        if shutil.disk_usage(root).free >= free_target:
            break
        if _in_use(entry):
            continue
        size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        shutil.rmtree(entry, ignore_errors=True)
        freed += size
    return freed


def _publish_directory(
    cache: CacheConfig,
    key: PurePosixPath,
    *,
    build: Callable[[Path], None],
    scratch_root: Path | None,
) -> Path:
    """Build/publish one directory artifact and return the **path** it lives at.

    `relarena.cache.cached_artifact` is the shared layer for this, and it does
    not fit: it returns a *value* that `load` materialized in memory, and builds
    a miss inside a `TemporaryDirectory` that is deleted before it returns. Both
    are right for a feature matrix and wrong for this artifact, whose consumer
    (`rt`) is handed a directory and reads it itself, over the whole life of a
    fit. So the mechanics are reproduced here, for this artifact only, rather
    than by changing a shared contract every other model already depends on.

    What is reproduced is exactly `cached_artifact`'s discipline, so the store
    stays one kind of thing: keys are relative POSIX paths beneath the cache
    root, a miss is built at a unique staging path and renamed into place
    atomically, a loser of a publish race discards its own tree rather than
    merging into the winner's, and a hit creates no locks or temporary files.

    Miss policy follows `cache.on_miss`: `raise` for configured benchmark runs
    (a miss is a warming mistake, not a licence to spend an hour re-embedding a
    database), `fill` for the warmer. Unconfigured or `compute` builds beneath
    `scratch_root`, which the **caller owns and must delete** — the path has to
    outlive this call, so it cannot be a `TemporaryDirectory` here.
    """
    parts = key.parts

    def done(path: Path) -> bool:
        return path.is_dir()

    if cache.directory is None or cache.on_miss == "compute":
        if scratch_root is None:
            raise ValueError(
                "RT preprocessing needs a caller-owned scratch_root when no "
                "persistent store is configured: the built directory has to "
                "outlive this call."
            )
        path = Path(scratch_root).joinpath(*parts)
        if done(path):
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        build(path)
        return path

    destination = Path(cache.directory).joinpath(*parts)
    if done(destination):
        _claim(destination)
        return destination
    if cache.on_miss == "raise":
        raise CacheMiss(
            f"cache miss for {str(key)!r} in {cache.directory}. {WARM_HINT}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Make room before writing, not after failing to: the disk that filled had
    # 6G left and the export needed 40.
    root = Path(cache.directory)
    if shutil.disk_usage(root).free < _REAP_FREE_BYTES:
        freed = reap(root)
        if freed:
            logger.info("rt: reaped %.1fG of unused exports", freed / 1024**3)
    staging = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        build(staging)
        try:
            staging.rename(destination)
        except OSError as exc:
            # Another builder got there first. The key pins the inputs, so its
            # tree is ours by construction; drop ours rather than merge.
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not done(
                destination
            ):
                raise
        _claim(destination)
        return destination
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _embed(
    pre_dataset_dir: Path, embedder: str, batch_size: int, cache: CacheConfig
) -> None:
    """Fill in `text_emb_<embedder>.bin`, reusing an identical one if there is one.

    The embedding pass is the expensive half of an export — a
    sentence-transformer over every text cell of the database, ~5 h per job on
    rel-amazon's 19.5M texts — and the exports of one run repeat it for text
    that is *byte-identical*. Three exports are built per task (the selection
    arm's, the refit's, and the one test is scored beside), and the reporting
    arm's two differ only by a test table whose columns are an id, a timestamp
    and a target: no text at all. So `text.json` is the same file in both. It is
    also the same across tasks of one dataset that share a split horizon, since
    the database rows the harness censors are then the same rows.

    Rather than reason about which of those cases hold, this keys a store on the
    **content** of `text.json`. Identical text gives identical embeddings, so a
    hit is exact by construction and needs no argument about how it arose; a
    miss just embeds. The entry is hardlinked where the filesystem allows it, so
    N exports of one database cost one copy of the largest file in them.

    Skipped when no persistent store is configured — there is nowhere shared to
    put it, and a scratch export is a one-off anyway.
    """
    from rt.preprocess import embed_dataset, update_meta_with_embeddings

    text = pre_dataset_dir / "text.json"
    root = None if cache.directory is None else Path(cache.directory) / "rt" / "emb"
    entry = None
    if root is not None and text.exists():
        digest = hashlib.sha256(text.read_bytes()).hexdigest()
        entry = root / embedder / f"{digest}.bin"

    target = pre_dataset_dir / f"text_emb_{embedder}.bin"
    if entry is not None and entry.exists():
        try:
            os.link(entry, target)
        except OSError:  # cross-device, or no hardlink support
            shutil.copyfile(entry, target)
        num_text = len(json.loads(text.read_text()))
        # bfloat16, row-major (num_text, d_text) -- `embed_dataset`'s own sum.
        d_text = target.stat().st_size // (max(num_text, 1) * 2)
        update_meta_with_embeddings(pre_dataset_dir, embedder, d_text)
        logger.info("rt: reused text embeddings from %s", entry)
        return

    d_text = embed_dataset(pre_dataset_dir, embedder, batch_size)
    update_meta_with_embeddings(pre_dataset_dir, embedder, d_text)
    if entry is None:
        return
    # Publish for the next export, by the same atomic-rename discipline the
    # directory store uses: a loser of the race drops its own copy.
    entry.parent.mkdir(parents=True, exist_ok=True)
    staging = entry.with_name(f".{entry.name}.tmp-{uuid.uuid4().hex}")
    try:
        shutil.copyfile(target, staging)
        os.replace(staging, entry)
    except OSError as exc:  # a full disk here must not fail the export
        logger.warning("rt: could not publish text embeddings to %s: %s", entry, exc)
        staging.unlink(missing_ok=True)


def preprocessed_dir(
    db: Database,
    task: EntityTask,
    splits: dict[str, Table],
    *,
    cache: CacheConfig,
    identity: RunIdentity | None,
    scratch_root: Path | None = None,
    db_name: str = "relarena",
) -> Path:
    """Return the directory of RT tensors for `db` + `splits`, building if needed.

    The returned path is a `pre_dir` in RT's sense: it holds one subdirectory
    per database, here just `db_name`. Treat it as read-only — it may be a
    shared cache entry another process is also reading.
    """

    def build(destination: Path) -> None:
        dataset_dir = destination / "_dataset"
        _write_dataset_dir(dataset_dir, db_name, db, task, splits)
        pre_dir = destination / "pre"
        pre_dir.mkdir(parents=True, exist_ok=True)

        from rt.preprocess import one

        # Rustler first, embedding second, so the text list exists to be looked
        # up before an hour is spent recomputing embeddings for it.
        args = preprocess_args(
            dataset=str(dataset_dir), out_dir=str(pre_dir), embed=False
        )
        one(**args)
        _embed(pre_dir / db_name, args["embedder"], args["batch_size"], cache)
        # The parquet source is only an input to rustler; keeping it would
        # roughly double a published artifact for nothing.
        shutil.rmtree(dataset_dir)

    root = _publish_directory(
        cache,
        _key(db, task, splits, identity),
        build=build,
        scratch_root=scratch_root,
    )
    return root / "pre"


def target_stats(train_table: Table, task: EntityTask) -> tuple[float, float]:
    """`(mean, std)` of the training target — RT's regression normalizer.

    RT emits regression predictions in normalized units and upstream
    denormalizes them as `pred * std + mean` with the **training** split's mean
    and standard deviation (`ddof=1`, a zero standard deviation read as 1). This
    reproduces that from the table the model was actually trained on, which on
    the refit split is the train+val union.
    """
    values = pd.to_numeric(train_table.df[task.target_col], errors="coerce").dropna()
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    return mean, (std if std not in (0.0,) and pd.notna(std) else 1.0)
