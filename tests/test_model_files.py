"""Tests for the pretrained-model files downloaded on first use."""

import json
import shutil

import pytest

from mlgw_bns import model_files


def test_manifest_matches_the_packaged_files():
    """After a retrain, the manifest must be written again (and the files
    published): a stale one would download other regressors."""
    manifest = model_files.read_manifest("default_hom")
    for name, expected in manifest["files"].items():
        path = model_files._data() / name
        if path.exists():
            assert model_files.sha256(path) == expected, name


@pytest.fixture
def unpackaged(tmp_path, monkeypatch):
    """A package without the regressors, whose manifest points to a
    local copy of them, and an empty cache."""
    published = tmp_path / "published"
    published.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    manifest = model_files.read_manifest("default_hom")
    name = sorted(manifest["files"])[0]
    shutil.copy(model_files._data() / name, published / name)
    files = {name: manifest["files"][name]}
    (data / "default_hom_files.json").write_text(
        json.dumps({"url": published.as_uri() + "/", "files": files})
    )
    monkeypatch.setattr(model_files, "_data", lambda: data)
    monkeypatch.setenv("MLGW_BNS_CACHE", str(tmp_path / "cache"))
    return name, files[name], published


def test_download_once_into_the_cache(unpackaged):
    name, expected, published = unpackaged
    path = model_files.model_file("default_hom", f"data/{name}")
    assert path.parent == model_files.cache_directory()
    assert model_files.sha256(path) == expected
    # the second time, from the cache
    (published / name).unlink()
    assert model_files.model_file("default_hom", f"data/{name}") == path


def test_download_checks_the_hash(unpackaged):
    name, _, published = unpackaged
    (published / name).write_bytes(b"not the regressor")
    with pytest.raises(OSError, match="SHA-256"):
        model_files.model_file("default_hom", f"data/{name}")
    assert not any(model_files.cache_directory().iterdir())
