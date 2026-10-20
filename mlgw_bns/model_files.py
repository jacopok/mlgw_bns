r"""The heavy files of the pretrained models, downloaded on first use.

The regressors of a pretrained model (its ``*_nn.pkl`` files, tens of MB)
are not in the package: a manifest that is,
``data/<model>_files.json``, gives the URL they are published at and the
SHA-256 of each. :func:`model_file` returns the file in the package if it is
there (a checkout of the repository, where they are kept in Git LFS), and
otherwise the copy in the cache,
downloading and checking it the first time. The cache is
``$MLGW_BNS_CACHE`` if set, else ``$XDG_CACHE_HOME/mlgw_bns`` (by default
``~/.cache/mlgw_bns``); its files are named by their hash, so the files of
different releases do not collide.

On machines without internet access (such as the compute nodes of a
cluster), download them beforehand on one that has it, into a cache the
others see::

    MLGW_BNS_CACHE=/shared/mlgw_bns python -m mlgw_bns.model_files download default_hom

To publish a new model: write its manifest from the files in ``data/``,
then upload those files where it says (here, the assets of a GitHub
release)::

    python -m mlgw_bns.model_files manifest default_hom \
        --url https://github.com/jacopok/mlgw_bns/releases/download/default_hom-v2.0.0/
    gh release create default_hom-v2.0.0 mlgw_bns/data/default_hom*_nn.pkl
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Optional

#: the files of a model that are downloaded instead of shipped
HEAVY_PATTERN = "{model}_l[0-9]_m[0-9]_nn.pkl"


def _data() -> Path:
    return Path(str(files(__package__).joinpath("data")))


def manifest_path(model_name: str) -> Path:
    return _data() / f"{model_name}_files.json"


def read_manifest(model_name: str) -> dict:
    """``{"url": base URL, "files": {name: sha256}}`` of a model."""
    with open(manifest_path(model_name)) as file:
        return json.load(file)


def cache_directory() -> Path:
    if "MLGW_BNS_CACHE" in os.environ:
        return Path(os.environ["MLGW_BNS_CACHE"])
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "mlgw_bns"


def sha256(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, target: Path, expected: str) -> None:
    """``url`` to ``target``, atomically, if its hash is ``expected``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(handle, "wb") as file, urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, file)
        found = sha256(temporary)
        if found != expected:
            raise OSError(f"{url}: SHA-256 {found}, expected {expected}")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


#: how a file kept in Git LFS starts in a checkout made without git-lfs
LFS_POINTER = b"version https://git-lfs.github.com/spec/"


def _is_packaged(path: Path) -> bool:
    """Whether ``path`` is there, and not a Git LFS pointer in its place."""
    if not path.exists():
        return False
    with open(path, "rb") as file:
        return file.read(len(LFS_POINTER)) != LFS_POINTER


def model_file(model_name: str, filename: str) -> Path:
    """The path of ``filename`` (as ``data/default_hom_l2_m2_nn.pkl``) of a
    pretrained model: in the package if it is there (and not a Git LFS
    pointer, in a checkout without git-lfs), else in the cache, downloaded
    the first time."""
    packaged = _data() / Path(filename).name
    if _is_packaged(packaged):
        return packaged
    manifest = read_manifest(model_name)
    name = packaged.name
    if name not in manifest["files"]:
        raise FileNotFoundError(f"{name} is neither in the package nor in the manifest of {model_name}")
    expected = manifest["files"][name]
    cached = cache_directory() / f"{expected[:16]}_{name}"
    if not cached.exists():
        url = manifest["url"] + name
        print(f"mlgw_bns: downloading {url} to {cached.parent} (once)")
        _download(url, cached, expected)
    return cached


def download(model_name: str) -> list:
    """Every downloadable file of a model, into the cache (if not in the package)."""
    return [model_file(model_name, name) for name in read_manifest(model_name)["files"]]


def write_manifest(model_name: str, url: str) -> Path:
    """The manifest of the heavy files of a model now in ``data/``, to be
    published at ``url`` (ending with ``/``)."""
    names = sorted(glob.glob(str(_data() / HEAVY_PATTERN.format(model=model_name))))
    if not names:
        raise FileNotFoundError(f"no {HEAVY_PATTERN.format(model=model_name)} in {_data()}")
    if not url.endswith("/"):
        url += "/"
    manifest = {"url": url, "files": {os.path.basename(name): sha256(name) for name in names}}
    path = manifest_path(model_name)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    command = commands.add_parser("download", help="fill the cache with the files of a model")
    command.add_argument("model", nargs="?", default="default_hom")
    command.set_defaults(run=lambda args: print("\n".join(map(str, download(args.model)))))
    command = commands.add_parser("manifest", help="write the manifest of the files of a model in data/")
    command.add_argument("model")
    command.add_argument("--url", required=True, help="where the files will be published")
    command.set_defaults(run=lambda args: print(write_manifest(args.model, args.url)))
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
