"""Task sources and censored temporal splits for supplied relational data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

import pandas as pd
from relbench.base import Database, Dataset, EntityTask, Table

from relarena_core.identity import RunIdentity
from relarena_core.metrics import primary_metric


def clean_database(
    db: Database, excluded_columns: dict[str, tuple[str, ...]] | None = None
) -> Database:
    """Drop empty/artifact columns and any explicitly excluded columns."""
    leaking = excluded_columns or {}
    table_dict = dict(db.table_dict)
    changed = False
    for table_name, table in db.table_dict.items():
        df = table.df
        # "Unnamed: N" columns are leftover pandas row indices from a CSV export.
        drop = {c for c in df.columns if c.startswith("Unnamed:")}
        if not df.empty:
            drop |= {c for c in df.columns if df[c].isna().all()}
        drop |= {c for c in leaking.get(table_name, ()) if c in df.columns}
        if not drop:
            continue
        changed = True
        table_dict[table_name] = Table(
            df=df.drop(columns=list(drop)),
            fkey_col_to_pkey_table=dict(table.fkey_col_to_pkey_table),
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )
    return Database(table_dict) if changed else db


def clean_task_table(
    task: EntityTask, table: Table, dataset_name: str, *, allow_extra: bool = False
) -> Table:
    """Validate task columns and optionally remove non-task columns."""
    canonical = {task.time_col, task.entity_col, task.target_col}
    extra = [c for c in table.df.columns if c not in canonical]
    if not extra:
        return table
    if not allow_extra:
        raise ValueError(
            f"Task table for {dataset_name!r} has unexpected non-canonical "
            f"column(s) {extra}; investigate rather than dropping."
        )
    keep = [c for c in table.df.columns if c in canonical]
    return Table(
        df=table.df[keep],
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def concat_tables(a: Table, b: Table) -> Table:
    """Concatenate two task label tables (e.g. train + val) into one, for refitting.

    Assumes both share the same schema (entity/foreign keys, time and target
    columns) — which holds for the splits of a single task. Inputs are untouched.
    """
    return Table(
        df=pd.concat([a.df, b.df], ignore_index=True),
        fkey_col_to_pkey_table=dict(a.fkey_col_to_pkey_table),
        pkey_col=a.pkey_col,
        time_col=a.time_col,
    )


@dataclass(frozen=True)
class Split:
    """Fields common to one fit→evaluate phase of nested temporal validation.

    Bundles the censored database with the label tables a phase needs, so a model
    can never accidentally see data past its phase's cutoff. Not instantiated
    directly: a phase is always an `InnerSplit` or an `OuterSplit`,
    which differ in *how the predictions are scored* (see those classes). The
    harness decides how each field is used (it does not pass `eval_table` to
    `fit` on the outer split, for instance) — see
    the tuning and final-fitting callers.
    """

    #: Database censored at `cutoff` — the only DB the model may read.
    db_state: Database
    #: The censoring cutoff (`val_timestamp` for inner, `test_timestamp` for outer).
    cutoff: pd.Timestamp
    #: The label table the model trains on.
    train_table: Table
    #: The table predictions are made on. For the outer split this is the *masked*
    #: test table — RelBench's `get_table("test")` strips the target column so the
    #: model can predict on the test entities/timestamps without seeing the answers
    #: (leakage prevention); the labels live only inside RelBench. The inner split's
    #: val table is unmasked, but the model is still expected to use it only as a
    #: prediction/early-stopping set, never to read its labels for training.
    eval_table: Table


@dataclass(frozen=True)
class InnerSplit(Split):
    """Tuning phase: fit `train` → score `val`.

    Scoring needs the val labels handed in explicitly. RelBench's
    `EntityTask.evaluate(pred, target_table=None)` treats `None` as "score
    against the *test* table" — there is no shortcut for val — so to score on val
    we must pass the val target table ourselves (`eval_target`). The val
    table's labels are not hidden, so this is just the val table itself.
    """

    name: ClassVar[str] = "inner"
    #: Val labels to score `eval_table` predictions against. Passed to
    #: `EntityTask.evaluate(pred, target_table=eval_target)` — required because
    #: `evaluate`'s `None` default would score against test, not val.
    eval_target: Table


@dataclass(frozen=True)
class OuterSplit(Split):
    """Final phase: fit the selected config → score `test`.

    `train_table` is the train-only table and `val_table` the
    val table, exposed separately so the harness can serve either final-fit regime:
    refit on their union (`refit_on_full_data=True`), or train on train alone with
    val as a held-out checkpoint/early-stopping set (`refit_on_full_data=False`).

    Deliberately carries *no* eval target. The test labels are hidden (the model's
    `eval_table` is the masked test table), and we score by calling
    `EntityTask.evaluate(pred, target_table=None)`, which makes RelBench load its
    own held-out test labels. Not materializing those labels into this object keeps
    the answer key out of every structure the harness passes around the model — so
    test-label leakage is impossible by construction, not just by convention.
    """

    name: ClassVar[str] = "outer"
    #: The val label table, held out from `train_table`. The harness
    #: unions it with the train table to refit on full data, or passes it as the
    #: monitoring set when a model trains on train alone.
    val_table: Table


def _drop_dangling_seeds(table: Table, entity_col: str, num_entities: int) -> Table:
    """Drop seeds whose entity is absent from a censored entity table.

    `entity_col` is the seed table's column of entity ids (each a row index into the
    entity table); a value `>= num_entities` references an entity created after the
    val cutoff, absent from the censored graph. This is the val-cutoff analog of
    relbench's `EntityTask.filter_dangling_entities` (which only filters against the
    test-censored `get_db` count).
    """
    keep = table.df[entity_col] < num_entities
    if keep.all():
        return table
    return Table(
        df=table.df[keep].reset_index(drop=True),
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def _copy_timeless_tables(db: Database) -> None:
    """Replace each timeless table's DataFrame with a copy, in place.

    `Table.upto` returns timeless tables (`time_col is None`) as the *same
    object* held by the source DB, so a censored DB shares their DataFrames with
    it. Any in-place mutation of the censored DB (e.g. the dangling-FK scrub in
    `validate_and_correct_db`, `df.loc[mask, fkey] = None`) would otherwise
    leak back into the source DB and corrupt later splits derived from it.
    """
    for name, table in db.table_dict.items():
        if table.time_col is None:
            db.table_dict[name] = Table(
                df=table.df.copy(),
                fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                pkey_col=table.pkey_col,
                time_col=table.time_col,
            )


class TaskSource:
    """Construct temporal splits from supplied database and task objects."""

    @staticmethod
    def _prepare_db(db: Database, dataset_name: str) -> Database:
        return clean_database(db)

    @staticmethod
    def _prepare_table(task: EntityTask, table: Table, dataset_name: str) -> Table:
        return clean_task_table(task, table, dataset_name)

    @classmethod
    def from_objects(
        cls,
        dataset: Dataset,
        task: EntityTask,
        *,
        dataset_name: str = "user",
        task_name: str | None = None,
        run_identity: RunIdentity | None = None,
    ) -> TaskSource:
        """Construct censored splits from supplied dataset and task objects.

        The default preparation drops empty and artifact database columns and
        validates task columns. Subclasses can apply source-specific cleaning
        through the preparation hooks. Names scope identity metadata.
        """
        self = cls.__new__(cls)
        self.dataset_name = dataset_name
        self.task_name = task_name if task_name is not None else type(task).__name__
        self._identity = run_identity or RunIdentity(
            dataset_name, None, self.task_name, None
        )
        self._dataset = dataset
        self._task = task
        self._db = cls._prepare_db(self._dataset.get_db(), dataset_name)
        self._tables = {
            split: cls._prepare_table(
                self._task, self._task.get_table(split), self.dataset_name
            )
            for split in ("train", "val", "test")
        }
        return self

    def run_identity(self, phase: str | None = None) -> RunIdentity:
        """Return optional source metadata scoped to one run phase."""
        return self._identity.for_phase(phase)

    @property
    def task(self) -> EntityTask:
        """The underlying RelBench task (defines the target, metrics, `evaluate`)."""
        return self._task

    @property
    def metric(self) -> Callable[..., float]:
        """The primary metric this task is tuned and selected on (by task type)."""
        return primary_metric(self._task)

    def inner_split(self) -> InnerSplit:
        """Tuning split: fit `train` → score `val`, DB frozen at `val_timestamp`.

        Censoring the DB at the val cutoff makes validation features *frozen at
        their cutoff*, mirroring how test features are frozen at the test cutoff —
        see `docs/temporal-validation.md` for why this matters for aggregating
        models. `eval_target` is the val table itself (its labels are not hidden).
        """
        # The val-cutoff removes pkey rows dated after val_timestamp; any foreign key
        # pointing *forward* in time to such a row (e.g. an attendance row referencing
        # a later event) is now dangling. get_db() scrubs dangling FKs for the test
        # cutoff via validate_and_correct_db, but this extra censor needs the same
        # correction or make_pkey_fkey_graph asserts on the out-of-range index.
        inner_db = self._db.upto(self._dataset.val_timestamp)
        # `validate_and_correct_db` scrubs dangling FKs in place, so decouple the
        # timeless tables `upto` shares with `self._db` first (see the helper),
        # otherwise the scrub leaks back into `self._db` and corrupts outer_split.
        _copy_timeless_tables(inner_db)
        self._dataset.validate_and_correct_db(inner_db)
        # The same val-cutoff can shrink the entity table below seeds that reference
        # entities created after val_timestamp — relbench's get_table only filtered
        # seeds against the *test* count. Drop those seeds here (the val-cutoff analog
        # of EntityTask.filter_dangling_entities) so a graph sampler can't index past
        # the censored entity table. The val table is filtered once and used for both
        # eval_table and eval_target, so scored predictions still align with targets.
        n_entities = len(inner_db.table_dict[self._task.entity_table].df)
        train_table = _drop_dangling_seeds(
            self._tables["train"], self._task.entity_col, n_entities
        )
        val_table = _drop_dangling_seeds(
            self._tables["val"], self._task.entity_col, n_entities
        )
        return InnerSplit(
            db_state=inner_db,
            cutoff=self._dataset.val_timestamp,
            train_table=train_table,
            eval_table=val_table,
            eval_target=val_table,
        )

    def outer_split(self) -> OuterSplit:
        """Final split: score `test` after fitting, DB at `test_timestamp`.

        Exposes the train and val tables separately (rather than pre-unioning them)
        so the harness can serve either final-fit regime — refit on their union, or
        train on train alone with val held out. No `eval_target`: the test labels
        are hidden, so scoring goes through `EntityTask.evaluate(pred,
        target_table=None)` and RelBench supplies them.
        """
        return OuterSplit(
            # No-op re-censor: get_db() already censors the DB at test_timestamp
            # (upto_test_timestamp=True), so this .upto() drops nothing. Kept for
            # readability/symmetry with inner_split's explicit val-cutoff censoring.
            db_state=self._db.upto(self._dataset.test_timestamp),
            cutoff=self._dataset.test_timestamp,
            train_table=self._tables["train"],
            val_table=self._tables["val"],
            eval_table=self._tables["test"],
        )
