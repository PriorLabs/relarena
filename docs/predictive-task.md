# Defining a predictive task with RPI

The **Relational Predictive Interface (RPI)** lets you define an entity-level
prediction task—binary classification or regression—over your own relational
database and run it with the models registered in RelArena. `PredictiveQuery` is
the Python façade for this interface.

## What even is a predictive task over a relational database?

How to answer this well remains an open problem in the field. Task types in the
literature are numerous and models generally don't transfer between them - even
within the same "task type," benchmark and method differences can substantially
affect performance. We're developing a taxonomy to help clarify this landscape
for the forthcoming release report.

We currently support **entity-level forecasting tasks**, which cover most of the
tasks in [RelBench v1](https://arxiv.org/abs/2407.20060) (excluding its
recommendation tasks). **Entity-level** means predictions attach to the rows of
one table: you pick an entity table - drivers, customers, sellers - and the task
asks, for each row at a given time, a question about that entity's near future
(that is the forecasting part): *will this driver fail to finish a race in the
next 30 days; will this seller receive no orders in the next 30 days*. The label
isn't a column in your data - a SQL query computes it from the database's own
future rows, so any relational database plus a question of that shape becomes a
supervised prediction task. Two task types are supported today:
`binary_classification` (0/1) and `regression` (numeric).

A task is two YAML files: a **task file** (the label SQL, split timestamps, and what
to predict) and a **database file** (the schema and paths to CSV or Parquet tables)
that the task references—so one database file can back many tasks. Load and run it:

```python
from relarena.userdb import PredictiveQuery, PredictiveQuerySpec

spec = PredictiveQuerySpec.from_yaml("task.yaml", data_dir="data/")
preds = PredictiveQuery(spec).fit(model="tabpfn-rel-client").predict()
```

`from_yaml` reads the task file, resolves its `database:` reference (a path relative
to the task file), and loads the database. `fit` builds the dataset, then tunes and
fits the model on history; `predict` scores the label-less rows at the end of the
data.

## Build the task in four steps

1. **Map the database** (in the database file). Per table: its data file, primary
   key (`pkey`), time column
   (`time_col`, event/log tables only), and foreign keys (`fkeys`:
   `column -> table-it-points-to`).
2. **Pick the entity and target.** Choose the `entity_table` you're predicting
   over - it must have a `pkey`, and `entity_col` carries that primary key (a
   foreign key to `entity_table.pkey`). Decide what you're predicting: `target_col`
   is the label the query *computes* over the forward window (it isn't a column in
   your data), and `task_type` is `binary_classification` (0/1) or `regression`
   (numeric). Steps 3-4 are where you write that computation and choose the windows.
3. **Write the label SQL** (rules below).
4. **Choose split timestamps** (`val_timestamp` / `test_timestamp`): rows before
   `val` train, between `val` and `test` validate, after `test` test. A good default
   is to keep val and test each **one `timedelta` wide** (`num_eval_timestamps: 1`)
   and set `test_timestamp` roughly one `timedelta` after `val_timestamp`, as
   RelBench does. RelArena freezes the validation and test databases at their
   respective phase boundaries, so if the data has a strong temporal component
   the model's view goes stale fast and predictions far past the cutoff tend to
   decay toward the constant baseline—a one-window-ahead
   horizon is what the model can most reliably generalize to. Widen the windows or
   the gap if your task genuinely calls for it, but expect that trade-off.

   These cutoffs belong to the **task**, not the data source. In particular,
   `materialize_relbench(...)` exports the full RelBench database, including rows
   after its original test cutoff. The bundled RelBench task YAMLs retain the
   original timestamps to reproduce the benchmark, but a copied task YAML can set
   different `val_timestamp` and `test_timestamp` values over the same tables.

The two files' shapes are defined by
[`database.schema.json`](../src/relarena/userdb/database.schema.json) and
[`task.schema.json`](../src/relarena/userdb/task.schema.json) — JSON Schemas with a
description on every field, validated on load, so a malformed file fails fast with a
pointer to the offending field rather than an opaque error later.

## Map the database

The **database file** (`db.yaml` here) — table name -> schema, reusable across tasks:

```yaml
drivers:
  pkey: driverId               # static dimension table: pkey only, no time_col
  path: drivers.parquet        # optional; defaults to "<table-name>.parquet" under data_dir
results:
  pkey: resultId
  time_col: date               # event table: has a time column
  fkeys:
    driverId: drivers          # results.driverId references drivers.<pkey>
    raceId: races
```

### Schema rules & gotchas

- Primary keys are reindexed to `0..n-1` internally, so string or gappy ids are fine
  - but each `pkey` must be unique and non-null.
- Every `fkeys` value must name another table in the database file, and that table
  must declare the `pkey`.
- Only event/log tables get a `time_col`; static dimension tables omit it. A
  `time_col`-less event table is treated as static and leaks future rows into the
  features, so derive a real event-time column when the source lacks one.

## Writing the label query

### The task file

The **task file** — references the database, defines the label and the split. The
`query` is the crux; how it works (`timestamp_df`, the forward window, the column
rules) is broken down below:

```yaml
database: db.yaml              # path to the database file, relative to this task file
entity_table: drivers
entity_col: driverId
time_col: date                 # name of the time column the query emits
target_col: did_not_finish
task_type: binary_classification
timedelta: 30 days             # forward prediction window
num_eval_timestamps: 40        # how many anchor times to spread across history
val_timestamp: '2005-01-01'
test_timestamp: '2010-01-01'
entities: all                  # "all", or an explicit list of entity ids
query: |
  SELECT t.timestamp AS date, re.driverId AS driverId,
         MAX(CASE WHEN re.statusId != 1 THEN 1 ELSE 0 END) AS did_not_finish
  FROM timestamp_df t
  JOIN results re                -- inner join: score only drivers racing in the
    ON re.date > t.timestamp     -- window (a LEFT join would emit NULL-driver rows
    AND re.date <= t.timestamp + INTERVAL '{timedelta}'   -- for empty windows)
  GROUP BY t.timestamp, re.driverId
```

The files define the task only - the model and tuning settings aren't in them; you
choose those when you run it (see below).

### How the anchor grid (`timestamp_df`) works

You never build `timestamp_df` - RelBench does, and registers it before running
your query. It's a single `timestamp` column, one row per **anchor time**: it
supplies the *when*, and your join supplies the *who*. That's why every label query
starts `FROM timestamp_df t` and joins the entity or event tables to it - each
anchor `t` gets crossed with the entities you're scoring to produce one row per
(anchor, entity). The label comes from the forward window
`(t, t + INTERVAL '{timedelta}']`.

Label construction and feature visibility are separate. RPI constructs the
forward-looking label, while RelArena supplies a database censored at the current
validation or test phase boundary. Individual models decide whether their
historical features or neighborhoods are additionally restricted to each row's
anchor timestamp. Ignoring that finer cutoff cannot reveal test labels or advance
the database past the common phase boundary; see
[temporal-validation.md](temporal-validation.md).

The anchor times themselves come from the split boundaries and `timedelta`, and the
query runs once per split against that split's anchors:

- **train**: step backward from `val_timestamp` in `timedelta` steps down to the
  earliest data, so training reuses all of history (at least 3 anchors required).
- **val** / **test**: `num_eval_timestamps` steps forward from `val_timestamp` /
  `test_timestamp`. With `num_eval_timestamps: 1` that's a single anchor each.

### Rules

- **Emit exactly three columns**, aliased to `time_col`, `entity_col`, `target_col`
  (here `date`, `driverId`, `did_not_finish`). Nothing else - a query that returns
  any extra column is rejected, because a stray column would silently be fed to the
  model as an input feature (and, computed over the forward window, would leak).
- **Start from `timestamp_df t`** - the runner registers it with the anchor times in
  a column named `timestamp`. Join your event tables to it.
- **Window forward from the anchor:**
  `x.time > t.timestamp AND x.time <= t.timestamp + INTERVAL '{timedelta}'`. The
  label must come from this future window—that is what makes it a prediction, not
  a lookup. Keep the lower
  bound strict (`>`) so the anchor moment itself is excluded.
- **`{timedelta}` is the only placeholder.** It's substituted literally (a plain
  string replace), so write it bare - `INTERVAL '{timedelta}'` - and leave any other
  braces alone: DuckDB struct / MAP / JSON literals (`{'k': 1}`, `'{"k":1}'`) pass
  through untouched, no escaping needed. The value is already `30 days`, so `INTERVAL
  '{timedelta} days'` is wrong.
- **Table names must match the database file's table names exactly, case included.**
  A table keyed `AdsInfo` is `AdsInfo`, not `ads_info`.
- **The join sets the entity universe.** Take `entity_col` from the event table to
  score only entities active in the window (the DNF example). To score every entity,
  select `entity_col` from the entity table, `LEFT JOIN` the events, and default the
  target (`COALESCE(..., 0)`).

### Recurrence and churn

"Did the entity churn / repeat an action" is not a plain forward window - the label
also depends on the entity's past activity. Seed the at-risk population from a
**backward** window, label from the **forward** window:

```sql
SELECT timestamp, seller_id,
  CAST(NOT EXISTS (                                  -- churn = no activity ahead
    SELECT 1 FROM order_items WHERE order_items.seller_id = sellers.seller_id
      AND purchase_ts > timestamp AND purchase_ts <= timestamp + INTERVAL '{timedelta}'
  ) AS INTEGER) AS churn
FROM timestamp_df, sellers
WHERE EXISTS (                                       -- only sellers active recently
    SELECT 1 FROM order_items WHERE order_items.seller_id = sellers.seller_id
      AND purchase_ts > timestamp - INTERVAL '{timedelta}' AND purchase_ts <= timestamp
)
```

See `examples/olist_seller_churn.yaml` and the RelBench `*.user-churn` /
`rel-event.user-repeat` specs for the full pattern.

## Run it

### End to end

One pipeline: `fit` builds the dataset, tunes on train→val, and performs the
selected model's final-fit regime; `predict` scores label-less rows at
`test_timestamp` by default. A task may set `at_timestamp` to request a different
prediction anchor. Keep the `PredictiveQuery` around to reuse the fitted model or
inspect the tuning trials (`.trials` / `.config`).

Following the RelBench protocol, the feature database is frozen at the task's
`test_timestamp`. Setting a later `at_timestamp` changes the timestamp of the
prediction rows, but it does **not** expose database rows written after
`test_timestamp`; RelArena emits a warning when this happens. To use a later
database snapshot, define a task with a later `test_timestamp` and fit it under
that split instead.

**Choosing the model.** The model is a run-time argument, instead of being part of
the spec, which makes it possible to easily compare different models:

```python
spec = PredictiveQuerySpec.from_yaml("task.yaml", data_dir="data/")
for model in ["constant-global", "lightgbm", "tabpfn-rel-client"]:
    preds = PredictiveQuery(spec).fit(model, n_trials=10).predict()
```

Good starting points are `constant-global` (constant baseline), `lightgbm` (entity-only),
and the `tabpfn-rel` variants (cross-table features). RPI can run any compatible
registered model; the package README is the
[canonical model inventory](../README.md#models), including paper-facing and
local variants. End-to-end `RelArenaSystem` registrations target benchmark
runs and are not accepted here: RPI fits once and predicts later at a
caller-selected timestamp, which is a different lifecycle from a system's
single split-to-predictions `run`. `n_trials` controls the requested tuning
budget (`0` skips tuning and fits the default config); `seed` sets the RNG. Prefer
`n_trials=0` for API runs unless repeated hosted fits are intentional.

**Watch the graph fan-out.** RDBLearn and the `tabpfn-rel` variants build their features by
walking the foreign-key graph with fastdfs, and the cost grows fast with how many
tables an entity links out to. An entity with a wide fan-out - a football match that
links 20-odd lineup, team, and event tables is the pathological case - can blow the
DFS feature space up and make a run prohibitively slow or memory-hungry. If a run
hangs, prune the schema to the tables that plausibly carry signal or reduce the
maximum depth up to which fastdfs joins.

**Caching for large or repeated runs.** Built-in DFS methods accept a local
`cache_dir` through `fit` and `predict`, reusing matrices across tuning trials,
the final fit, and prediction. The first RPI run fills the local store and later
runs over the same inputs read it back. Nothing is uploaded. Omit `cache_dir` to
fall back to `RELARENA_CACHE_DIR`, or to compute without persistent caching when
neither is set. The underlying `relarena.cache` API is optional and experimental;
models may implement caching independently.

```python
pq = PredictiveQuery(spec).fit("tabpfn-rel-local", cache_dir="/scratch/my_db_cache")
preds = pq.predict()   # reuses the cache_dir passed to fit
```

When the source data contain the complete window after `test_timestamp`,
materialize those historical outcomes and join them to predictions for your own
evaluation. Test labels are never passed to the model:

```python
test_labels = pq.compute_test_labels()
```

By default, coverage is checked against the database's latest timestamp. Pass
`data_end_timestamp=...` when the database is known to be complete only through
a different date, such as for a partial or sparse extract. The method raises if
that cutoff does not cover all configured test label windows. A genuine
production forecast has no labels until its forward window has happened.

To split the expensive DFS build (CPU-bound, memory-heavy) from the GPU fit, precompute
the store first with `precompute_cache` on a big CPU node, then `fit` on the GPU reads
it instead of recomputing:

```python
pq = PredictiveQuery(spec)
pq.precompute_cache("/scratch/my_db_cache")                          # CPU, no TFM
pq.fit("tabpfn-rel-local", cache_dir="/scratch/my_db_cache")      # GPU, reads the store
preds = pq.predict()
```

`examples/olist_seller_churn.py` shows the full flow on real data (the Olist
worked example below); its header has the exact data-download and run commands.

### Does a model even help? Check a constant baseline

Fit the `constant-global` baseline every time and treat it as the bar to clear: a constant,
the median for regression or the majority class for classification
(`PredictiveQuery(spec).fit("constant-global")`). Where the entity's own history is
the obvious signal, also check a per-entity baseline (each entity's own past
average). Ship a model only if it clearly beats these.

Not every task you can formulate is better solved by a model. Whether relational
context helps is task-dependent and, honestly, still an open question. Olist churn
clears both bars (TabPFN-Rel 0.79 vs Constant (global) 0.50 and Constant (per-entity) 0.69);
plenty of reasonable-looking tasks won't, and the constant baseline is how you find out - not
intuition.

And when nothing beats the baselines, the result is genuinely ambiguous. It could be
that the interface fails to surface the signal (no obvious reason it would, but we
can't rule it out), or that the task as formulated just isn't predictable from the
data.
We can't currently tell those apart, so read a flat result as "no signal we can find
for this task as posed" rather than a verdict either way.

### Worked examples

- **RelBench v1 (21 tasks)** in `src/relarena/userdb/relbench_v1/`
  - one folder per dataset (a shared `db.yaml` + one file per task), reproducing
  RelBench's splits byte-for-byte. `materialize_relbench("rel-f1", "data/rel-f1")`
  writes the full tables to parquet; `relbench_v1_spec(dataset, task)` loads the
  benchmark's reference cutoffs. Copy the closest task YAML and change its split
  timestamps to define a new task over the materialized data.
- **Bring-your-own database** in `examples/`: an end-to-end run on
  the Kaggle Olist e-commerce data (`olist_seller_churn.yaml` task +
  `olist_database.yaml` schema + `olist_seller_churn.py`). Point the database file
  straight at the raw CSVs (per-table
  `columns` curate the features; only `order_items` needs a derived timestamp), fit
  -> predict -> evaluate. It predicts seller churn, and it is where relational
  context earns its keep: held-out ROC-AUC is `constant-global 0.50`, `lightgbm`
  (entity-only) `0.58`,
  `constant-per-entity 0.69`, `TabPFN-Rel 0.79` - the cross-table order/review history lifts
  TabPFN-Rel above every baseline, including each seller's own past churn rate.
