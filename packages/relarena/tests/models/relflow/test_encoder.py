"""Offline checks for the adapter's immutable frozen-text encoder snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from relarena.models.relflow import raw, search_space
from relarena_core.search_space import TaskStats


def structure(*, text: bool) -> raw.Task:
    return raw.Task(
        dataset="fixture",
        task="encoder",
        task_type="regression",
        entity_table="entities",
        entity_key="entity_id",
        anchor_entity_column="entity_id",
        target_column="label",
        root_time=None,
        columns=(raw.text("summary"), raw.text("description"))
        if text
        else (raw.number("amount"),),
        lookups=(),
        branches=(
            raw.Branch(
                name="events",
                table="events",
                on='{t}."entity_id" = {parent}."__entity_key"',
                time=None,
                length=8,
                columns=(raw.text("note"),) if text else (raw.number("value"),),
            ),
        ),
    )


@pytest.mark.parametrize("store", [False, True])
def test__schema__text_fields__resolve_one_exact_encoder_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: bool
) -> None:
    pytest.importorskip("relflow")
    import huggingface_hub

    from relarena.models.relflow.training import schema

    snapshot = str(tmp_path / "snapshot")
    calls = []

    def download(**options: Any) -> str:
        calls.append(options)
        return snapshot

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    config = search_space(TaskStats(3)).default_overrides
    configured = schema(
        structure(text=True), config, store=tmp_path / "text-store" if store else None
    )
    assert calls == [
        {
            "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "allow_patterns": [
                "config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
            ],
        }
    ]
    fields = configured["fields"]
    for field in (
        fields["summary"],
        fields["description"],
        fields["events"]["fields"]["note"],
    ):
        assert field["model"] == snapshot
        assert field["encoder_pooling"] == "mean"
        assert field["encoder_batch_size"] == 320
        assert field["max_length"] == 64
        assert field.get("store") == (str(tmp_path / "text-store") if store else None)


def test__schema__no_text__does_not_acquire_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("relflow")
    import huggingface_hub

    from relarena.models.relflow.training import schema

    def unexpected(**options: Any) -> str:
        raise AssertionError(f"No text field may acquire an encoder: {options!r}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected)
    configured = schema(
        structure(text=False), search_space(TaskStats(3)).default_overrides
    )
    assert configured["fields"]["amount"]["type"] == "number"


def test__schema__unknown_text_encoder__refuses_unpinned_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("relflow")
    import huggingface_hub

    from relarena.models.relflow.training import schema

    def unexpected(**options: Any) -> str:
        raise AssertionError(f"An unpinned encoder may not be acquired: {options!r}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected)
    monkeypatch.setitem(raw.TEXT, "model", "example/unpinned-encoder")
    with pytest.raises(ValueError, match="no frozen revision"):
        schema(structure(text=True), search_space(TaskStats(3)).default_overrides)
