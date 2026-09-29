"""Finding shot records in folders and zips, and locating the example datasets."""

import io
import zipfile
from pathlib import Path

import pytest

from shallowgeo import datasets
from shallowgeo.datasets import example_data, find_shot_files
from shallowgeo.refraction import load_shots

REPO = Path(__file__).resolve().parents[1]
SRT = REPO / "examples/data/2026-09-25-srt-line/raw"


@pytest.fixture
def line(tmp_path):
    """A folder of fake shot files plus the clutter real folders have."""
    d = tmp_path / "line"
    (d / "sub").mkdir(parents=True)
    for name in ("1001.sgy", "1002.SGY", "sub/1003.dat", "notes.txt", "._1001.sgy"):
        (d / name).write_bytes(b"x")
    return d


def _zip(folder: Path, dest: Path, *, finder_junk=True) -> Path:
    with zipfile.ZipFile(dest, "w") as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(folder.parent))
                if finder_junk:
                    z.writestr(f"__MACOSX/{p.relative_to(folder.parent).parent}/._{p.name}", b"")
    return dest


class TestFindShotFiles:
    def test_folder_is_searched_recursively_and_clutter_skipped(self, line):
        names = [p.name for p in find_shot_files(line)]
        assert names == ["1001.sgy", "1002.SGY", "1003.dat"]

    def test_zip_gives_the_same_files_without_finder_junk(self, line, tmp_path):
        z = _zip(line, tmp_path / "line.zip")
        names = sorted(p.name for p in find_shot_files(z, extract_to=tmp_path / "x"))
        assert names == ["1001.sgy", "1002.SGY", "1003.dat"]

    def test_glob_single_file_and_lists(self, line):
        assert [p.name for p in find_shot_files(str(line / "100*.sgy"))] == ["1001.sgy"]
        assert len(find_shot_files(line / "1002.SGY")) == 1
        assert len(find_shot_files([line / "1001.sgy", line / "sub"])) == 2

    def test_missing_and_empty_sources_say_what_was_looked_at(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            find_shot_files(tmp_path / "nope")
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no shot records"):
            find_shot_files(tmp_path / "empty")
        with pytest.raises(FileNotFoundError, match="no files match"):
            find_shot_files(str(tmp_path / "*.sgy"))


@pytest.mark.skipif(not SRT.exists(), reason="example dataset not present")
class TestLoadShotsFromZip:
    def test_zip_and_folder_load_the_same_shots(self, tmp_path):
        z = _zip(SRT, tmp_path / "srt.zip")
        assert list(load_shots(z)) == list(load_shots(SRT))
        assert len(load_shots(z)) == 16


class TestExampleData:
    def test_found_in_the_checkout(self):
        if not SRT.exists():
            pytest.skip("example dataset not present")
        assert example_data("2026-09-25-srt-line") == SRT

    def test_unknown_name_lists_the_choices(self):
        with pytest.raises(KeyError, match="2026-09-25-srt-line"):
            example_data("no-such-line")

    def test_no_download_when_told_not_to(self, monkeypatch, tmp_path):
        monkeypatch.setattr(datasets, "_local_examples", list)
        with pytest.raises(FileNotFoundError, match="download=True"):
            example_data("2026-09-25-srt-line", download=False, cache=tmp_path)

    def test_download_keeps_only_the_dataset_and_is_cached(self, monkeypatch, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            root = "shallow-geophysics-main/examples/data"
            z.writestr(f"{root}/2026-09-25-srt-line/raw/3000.sgy", b"segy")
            z.writestr(f"{root}/2026-09-25-srt-line/dataset.md", b"# md")
            z.writestr(f"{root}/2026-09-18-refraction-line/raw/4009.sgy", b"other")
            z.writestr("shallow-geophysics-main/README.md", b"readme")
        calls = []

        def fake_urlopen(url, timeout=None):
            calls.append(url)
            return io.BytesIO(buf.getvalue())

        monkeypatch.setattr(datasets, "_local_examples", list)
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        raw = example_data("2026-09-25-srt-line", cache=tmp_path)
        assert raw == tmp_path / "2026-09-25-srt-line" / "raw"
        assert (raw / "3000.sgy").read_bytes() == b"segy"
        assert (raw.parent / "dataset.md").exists()
        assert not (tmp_path / "2026-09-18-refraction-line").exists()
        example_data("2026-09-25-srt-line", cache=tmp_path)
        assert len(calls) == 1
