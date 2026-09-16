"""RelGNN-owned graph key, directory codec, validation, and warming."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from relbench.base import Database

from relarena.cache import CacheConfig, cache_key, cached_artifact
from relarena.checksums import database_checksum
from relarena.identity import RunIdentity
from relarena.models._shared.gnn.graph_cache import DBGraphCache

_GRAPH_CACHE_VERSION = 2
_TEXT_EMBED_BATCH_SIZE = 256
_COMPLETE = "_COMPLETE"


def graph_cache_key(db: Database, run_identity: RunIdentity | None) -> str:
    """Return the complete graph key, intentionally omitting task identity."""
    dataset = "direct" if run_identity is None else run_identity.dataset
    fingerprint = (
        None if run_identity is None else run_identity.dataset_fingerprint
    ) or f"{int(database_checksum(db)):016x}"
    phase = (
        "direct"
        if run_identity is None or run_identity.phase is None
        else run_identity.phase
    )
    segments = [
        "relgnn",
        f"graph-v{_GRAPH_CACHE_VERSION}",
        f"{dataset}@{fingerprint}",
        f"phase-{phase}",
    ]
    if run_identity is not None and run_identity.data_version is not None:
        digest = hashlib.blake2s(
            run_identity.data_version.encode(), digest_size=8
        ).hexdigest()
        segments.append(f"data-{digest}")
    if phase == "predict":
        # PredictiveQuery identities fingerprint the source schema/content version,
        # while the graph sees a view censored at the prediction anchor. Distinguish
        # those views without pulling unrelated task/model fields into the key.
        digest = hashlib.blake2s(
            str(db.max_timestamp).encode(), digest_size=8
        ).hexdigest()
        segments.append(f"cutoff-{digest}")
    return str(cache_key(*segments))


def _validate_graph(data: Any, col_stats: dict[str, Any]) -> None:
    """Reject materialized timestamps that would crash the frame encoder."""
    import torch_frame
    from torch_frame.data.stats import StatType

    for node_type in data.node_types:
        tf = data[node_type].tf
        ts_cols = tf.col_names_dict.get(torch_frame.stype.timestamp)
        if not ts_cols:
            continue
        year = tf.feat_dict[torch_frame.stype.timestamp][..., 0]
        for index, column in enumerate(ts_cols):
            min_year = col_stats[node_type][column][StatType.YEAR_RANGE][0]
            n_bad = int((year[:, index] < min_year).sum())
            if n_bad:
                raise ValueError(
                    f"corrupt graph: node {node_type!r} col {column!r} has "
                    f"{n_bad} row(s) with year < {min_year}; bump the graph "
                    "version and warm a valid artifact."
                )


def _materialize(db: Database, target: Path) -> tuple[Any, dict[str, Any]]:
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from torch_frame.config.text_embedder import TextEmbedderConfig

    from relarena.models._shared.gnn._vendor.gnn import GloveTextEmbedding

    return make_pkey_fkey_graph(
        db,
        col_to_stype_dict=get_stype_proposal(db),
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device="cpu"),
            batch_size=_TEXT_EMBED_BATCH_SIZE,
        ),
        cache_dir=str(target),
    )


def _load(db: Database, path: Path) -> tuple[Any, dict[str, Any]]:
    if not (path / _COMPLETE).is_file():
        raise ValueError(f"incomplete RelGNN graph artifact at {path}")
    result = _materialize(db, path)
    _validate_graph(*result)
    return result


def _build(db: Database, path: Path) -> tuple[Any, dict[str, Any]]:
    path.mkdir()
    result = _materialize(db, path)
    _validate_graph(*result)
    (path / _COMPLETE).write_text("relgnn graph complete\n")
    return result


_GRAPH_CACHE = DBGraphCache()


def load_graph(
    db: Database, cache: CacheConfig, run_identity: RunIdentity | None
) -> tuple[Any, dict[str, Any]]:
    """Load/build and memoize the RelGNN graph for one censored database."""
    key = graph_cache_key(db, run_identity)
    variant = (cache, key)
    return _GRAPH_CACHE.get(
        db,
        lambda: cached_artifact(
            cache,
            key,
            storage="directory",
            load=lambda path: _load(db, path),
            build=lambda path: _build(db, path),
            warm_hint="Run python -m relarena.models.relgnn.warm_cache.",
        ),
        variant=variant,
    )
