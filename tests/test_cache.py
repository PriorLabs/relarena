"""Tests for generic opaque-artifact cache mechanics."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier

import pytest

from relarena.cache import (
    CACHE_DIR_ENV,
    DISABLE_CACHE_ENV,
    LEGACY_DISABLE_CACHE_ENV,
    CacheConfig,
    CacheMiss,
    cache_key,
    cached_artifact,
    parse_local_cache_dir,
    resolve_cache_config,
)


def _load_text(path: Path) -> str:
    return path.read_text()


def _build_text(value: str) -> Callable[[Path], str]:
    def build(path: Path) -> str:
        path.write_text(value)
        return value

    return build


def test__cache_config__mutation__is_rejected(tmp_path: Path) -> None:
    config = CacheConfig(tmp_path, "fill")
    with pytest.raises(FrozenInstanceError):
        config.on_miss = "raise"  # type: ignore[misc]


def test__resolve_cache_config__explicit_and_environment__explicit_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_DIR_ENV, "/unused")
    explicit = tmp_path / "explicit"
    assert resolve_cache_config(explicit, on_miss="raise") == CacheConfig(
        explicit, "raise"
    )


def test__resolve_cache_config__disable_switch__removes_persistent_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CACHE_DIR_ENV, "/configured")
    monkeypatch.setenv(DISABLE_CACHE_ENV, "1")
    assert resolve_cache_config(None, on_miss="fill") == CacheConfig(None, "fill")


def test__resolve_cache_config__legacy_disable_switch__warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CACHE_DIR_ENV, "/configured")
    monkeypatch.setenv(LEGACY_DISABLE_CACHE_ENV, "1")
    with pytest.warns(FutureWarning, match=DISABLE_CACHE_ENV):
        config = resolve_cache_config(None, on_miss="raise")
    assert config == CacheConfig(None, "raise")


def test__parse_local_cache_dir__home_path__expands() -> None:
    assert parse_local_cache_dir("~/cache") == Path.home() / "cache"


@pytest.mark.parametrize("uri", ["gs://bucket/cache", "gs:/bucket/cache"])
def test__parse_local_cache_dir__remote_uri__raises(uri: str) -> None:
    with pytest.raises(ValueError, match="local path"):
        parse_local_cache_dir(uri)


def test__cache_config__invalid_policy__raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid cache miss policy"):
        CacheConfig(tmp_path, "refresh")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "key", ["", "/absolute.bin", "../escape.bin", "a/./b", "a//b", "a\\b"]
)
def test__cached_artifact__unsafe_key__raises(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match="safe relative"):
        cached_artifact(
            CacheConfig(tmp_path, "fill"),
            key,
            storage="file",
            load=_load_text,
            build=_build_text("new"),
        )


def test__cache_key__unsafe_segment__raises() -> None:
    with pytest.raises(ValueError, match="separator"):
        cache_key("model", "../artifact.bin")


def test__cached_artifact__symlink_escape__raises(tmp_path: Path) -> None:
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        cached_artifact(
            CacheConfig(root, "fill"),
            "link/item.bin",
            storage="file",
            load=_load_text,
            build=_build_text("unsafe"),
        )
    assert list(outside.iterdir()) == []


def test__cached_artifact__file_fill_then_hit__loads_published_value(
    tmp_path: Path,
) -> None:
    key = "example-v1/item.bin"
    filled = cached_artifact(
        CacheConfig(tmp_path, "fill"),
        key,
        storage="file",
        load=_load_text,
        build=_build_text("filled"),
    )
    assert filled == "filled"
    assert (tmp_path / key).read_text() == "filled"

    hit = cached_artifact(
        CacheConfig(tmp_path, "raise"),
        key,
        storage="file",
        load=_load_text,
        build=lambda path: pytest.fail("hit must not build"),
    )
    assert hit == "filled"


def test__cached_artifact__configured_miss__raises_without_writing(
    tmp_path: Path,
) -> None:
    with pytest.raises(CacheMiss, match="run the example warmer"):
        cached_artifact(
            CacheConfig(tmp_path, "raise"),
            "example-v1/item.bin",
            storage="file",
            load=_load_text,
            build=_build_text("unused"),
            warm_hint="run the example warmer",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "config", [CacheConfig(None, "raise"), CacheConfig(None, "fill")]
)
def test__cached_artifact__no_store__builds_in_cleaned_scratch(
    config: CacheConfig,
) -> None:
    built_path: Path | None = None

    def build(path: Path) -> str:
        nonlocal built_path
        built_path = path
        path.write_text("scratch")
        return "scratch"

    assert (
        cached_artifact(
            config,
            "example-v1/item.bin",
            storage="file",
            load=_load_text,
            build=build,
        )
        == "scratch"
    )
    assert built_path is not None and not built_path.exists()


def test__cached_artifact__compute_policy__does_not_touch_configured_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    cached_artifact(
        CacheConfig(root, "compute"),
        "example-v1/item.bin",
        storage="file",
        load=_load_text,
        build=_build_text("scratch"),
    )
    assert not root.exists()


def test__cached_artifact__directory_fill__returns_builder_value_without_reload(
    tmp_path: Path,
) -> None:
    def build(path: Path) -> dict[str, int]:
        path.mkdir()
        (path / "data.bin").write_bytes(b"data")
        return {"built": 1}

    value = cached_artifact(
        CacheConfig(tmp_path, "fill"),
        "example-v1/index",
        storage="directory",
        load=lambda path: pytest.fail("fill must return the built value"),
        build=build,
        validate=lambda result: result["built"],
    )
    assert value == {"built": 1}
    assert (tmp_path / "example-v1/index/data.bin").read_bytes() == b"data"


def test__cached_artifact__relocated_read_only_tree__is_a_write_free_hit(
    tmp_path: Path,
) -> None:
    source, relocated = tmp_path / "source", tmp_path / "relocated"
    cached_artifact(
        CacheConfig(source, "fill"),
        "example-v1/item.bin",
        storage="file",
        load=_load_text,
        build=_build_text("portable"),
    )
    shutil.copytree(source, relocated)
    before = sorted(path.relative_to(relocated) for path in relocated.rglob("*"))
    relocated.chmod(0o555)
    for path in relocated.rglob("*"):
        path.chmod(0o444 if path.is_file() else 0o555)
    try:
        value = cached_artifact(
            CacheConfig(relocated, "raise"),
            "example-v1/item.bin",
            storage="file",
            load=_load_text,
            build=lambda path: pytest.fail("hit must not build"),
        )
        after = sorted(path.relative_to(relocated) for path in relocated.rglob("*"))
    finally:
        relocated.chmod(0o755)
        for path in relocated.rglob("*"):
            path.chmod(0o644 if path.is_file() else 0o755)
    assert value == "portable"
    assert after == before


def test__cached_artifact__concurrent_directory_fills__publish_one_complete_hit(
    tmp_path: Path,
) -> None:
    barrier = Barrier(2)

    def fill(value: str) -> str:
        def build(path: Path) -> str:
            path.mkdir()
            (path / "data.bin").write_text(value)
            barrier.wait()
            return value

        return cached_artifact(
            CacheConfig(tmp_path, "fill"),
            "example-v1/index",
            storage="directory",
            load=lambda path: (path / "data.bin").read_text(),
            build=build,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(fill, ["a", "b"]))
    assert values == ["a", "b"]
    published = (tmp_path / "example-v1/index/data.bin").read_text()
    assert published in values
    assert not list(tmp_path.rglob("*.tmp-*"))
