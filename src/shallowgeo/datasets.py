"""Finding shot records: folders, zip files, and the example datasets.

Two jobs that every teaching notebook starts with.

:func:`find_shot_files` turns whatever the student has -- a folder, a
``.zip`` of a folder, one file, a glob, or a list of those -- into a sorted
list of shot-record paths. Zips are unpacked to a temporary folder, and the
``__MACOSX/`` and ``._name`` entries that macOS Finder puts into every zip
it makes are skipped.

:func:`example_data` returns the ``raw/`` folder of one of the datasets in
``examples/data/``. In a clone of the repository it is found in place;
anywhere else (Colab, a ``pip install`` from GitHub) the repository archive
is downloaded once and the dataset cached under ``~/.cache/shallowgeo``.
"""

from __future__ import annotations

import glob
import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

__all__ = ["SHOT_SUFFIXES", "EXAMPLES", "find_shot_files", "example_data"]

SHOT_SUFFIXES = frozenset({".dat", ".sgy", ".segy", ".sg2"})

EXAMPLES = {
    "2026-09-18-refraction-line": "13 shots from 8 positions into a 92 ft spread (refraction)",
    "2026-09-25-srt-line": "16 shots, every other geophone, 69 ft spread (tomography)",
}

REPO_ARCHIVE = "https://github.com/Maurer-GEMLab/shallow-geophysics/archive/refs/heads/main.zip"


def _is_junk(path: Path) -> bool:
    return path.name.startswith("._") or "__MACOSX" in path.parts


def _extract(zip_path: Path, dest: Path | None) -> Path:
    dest = Path(tempfile.mkdtemp(prefix=f"{zip_path.stem}-")) if dest is None else Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)  # zipfile strips absolute paths and ".." components
    return dest


def find_shot_files(source, *, extract_to: str | Path | None = None) -> list[Path]:
    """Every shot record in ``source``, sorted by path.

    Parameters
    ----------
    source
        A folder (searched recursively), a ``.zip`` file, a single shot
        file, a glob pattern such as ``"raw/40*.sgy"``, or a list of any of
        these. Shot records are files ending in ``.dat``, ``.sgy``,
        ``.segy`` or ``.sg2``.
    extract_to
        Where to unpack zip files. Default: a new temporary folder each.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist or holds no shot records -- with what
        was looked at, so the message says which option went wrong.
    """
    items = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    found: set[Path] = set()
    for item in items:
        text = str(item)
        path = Path(text).expanduser()
        if any(ch in text for ch in "*?[") and not path.exists():
            matches = [Path(p) for p in glob.glob(str(path), recursive=True)]
            if not matches:
                raise FileNotFoundError(f"no files match {text!r}")
            found.update(find_shot_files(matches, extract_to=extract_to))
            continue
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        if path.is_dir():
            found.update(p for p in path.rglob("*")
                         if p.is_file() and p.suffix.lower() in SHOT_SUFFIXES
                         and not _is_junk(p))
        elif path.suffix.lower() == ".zip":
            sub = None if extract_to is None else Path(extract_to) / path.stem
            found.update(find_shot_files(_extract(path, sub)))
        elif path.suffix.lower() in SHOT_SUFFIXES and not _is_junk(path):
            found.add(path)
    if not found:
        raise FileNotFoundError(
            f"no shot records ({', '.join(sorted(SHOT_SUFFIXES))}) in {source!s}")
    return sorted(found)


def _local_examples() -> list[Path]:
    """Places an ``examples/data`` folder may be, nearest first."""
    here = Path(__file__).resolve()
    roots = [here.parents[2]]  # src/shallowgeo/datasets.py -> repository root
    cwd = Path.cwd()
    roots += [cwd, *cwd.parents, cwd / "shallow-geophysics"]
    return [r / "examples" / "data" for r in roots]


def example_data(name: str = "2026-09-25-srt-line", *, download: bool = True,
                 cache: str | Path | None = None) -> Path:
    """The ``raw/`` folder of an example dataset, downloading it if needed.

    Parameters
    ----------
    name
        One of :data:`EXAMPLES`. Each has a ``dataset.md`` next to its
        ``raw/`` folder describing the survey and its known problems.
    download
        Fetch the repository archive from GitHub when the dataset is not
        found locally. The archive is a few megabytes; only the dataset's
        folder is kept.
    cache
        Where downloaded datasets live. Default ``~/.cache/shallowgeo``.
    """
    if name not in EXAMPLES:
        raise KeyError(f"no example dataset {name!r}; have {sorted(EXAMPLES)}")
    for base in _local_examples():
        raw = base / name / "raw"
        if raw.is_dir():
            return raw
    cache = Path(cache or Path.home() / ".cache" / "shallowgeo").expanduser()
    dest = cache / name
    if (dest / "raw").is_dir():
        return dest / "raw"
    if not download:
        raise FileNotFoundError(
            f"example {name!r} is not in this checkout or in {cache}; "
            "call example_data(..., download=True) to fetch it")

    from urllib.request import urlopen

    with urlopen(REPO_ARCHIVE, timeout=60) as response:  # fixed https URL
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    marker = f"/examples/data/{name}/"
    members = [m for m in archive.namelist() if marker in m and not m.endswith("/")]
    if not members:
        raise FileNotFoundError(f"{name!r} is not in the repository archive at {REPO_ARCHIVE}")
    cache.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=cache))
    for member in members:
        target = tmp / member.split(marker, 1)[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest / "raw"
