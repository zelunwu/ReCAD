"""Tests for recad.data.download (network-free where possible)."""

from __future__ import annotations

import pytest

from recad.data import download as dl


def test_manifest_contains_all_sources():
    names = set(dl.SOURCES)
    assert {"xco2air", "socat", "gshhg", "sst", "sss", "adt", "wspd", "bathymetry"} <= names
    # open sources must have concrete downloaders; others are auth-gated
    assert dl.SOURCES["socat"].kind == "target"
    assert dl.SOURCES["sss"].requires_auth


def test_manifest_log(tmp_path):
    root = tmp_path / "data" / "raw"
    dl.download_source.__globals__["DATA_ROOT"] = root  # not used by writer, but safe
    (root / "xco2air" / "noaa_gml_mbl").mkdir(parents=True)
    (root / "xco2air" / "noaa_gml_mbl" / "co2_mm_gl.txt").write_text("x", encoding="utf-8")
    manifest = dl.write_manifest_log(root)
    txt = manifest.read_text(encoding="utf-8")
    assert "## xco2air" in txt and "present" in txt
    assert "## sss" in txt and "not downloaded" in txt


def test_download_file_from_file_uri(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"\x00\x01\x02\x03" * 1000)
    dest = tmp_path / "dst" / "out.bin"
    dl.download_file(src.as_uri(), dest, show_progress=False)
    assert dest.read_bytes() == src.read_bytes()


def test_download_file_resume(tmp_path):
    """file:// never honours Range, so a resumed call rewrites the file."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"A" * 3_000_000)
    dest = tmp_path / "out.bin"
    dl.download_file(src.as_uri(), dest, show_progress=False)
    assert dest.read_bytes() == src.read_bytes()
    # second call with resume=True: file:// ignores Range and writes wholly,
    # but the final content must equal the source (idempotence).
    dl.download_file(src.as_uri(), dest, resume=True, show_progress=False)
    assert dest.read_bytes() == src.read_bytes()


def test_sha256_and_checksum_sidecar(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    side = dl.write_checksum(p)
    assert side.exists()
    assert dl.sha256_of(p) in side.read_text(encoding="utf-8")


def test_unzip(tmp_path):
    import zipfile

    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("dir/file.txt", "content")
    out = tmp_path / "out"
    dl.extract_archive(z, out)
    assert (out / "dir" / "file.txt").read_text(encoding="utf-8") == "content"


def test_extract_tar_gz(tmp_path):
    import io
    import tarfile

    tar = tmp_path / "a.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        data = b"gs data"
        info = tarfile.TarInfo("bin/file.bin")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    out = tmp_path / "out"
    dl.extract_archive(tar, out)
    assert (out / "bin" / "file.bin").read_bytes() == data


def test_dds_dimensions_parsing():
    dds = """
Dataset {
  Float64 time[time = 372];
  Float64 ylat[ylat = 721];
  Float64 xlon[xlon = 1440];
  Grid {
   Float32 coast_fco2_ave_weighted[time = 372][ylat = 721][xlon = 1440];
  } SOCAT_v2026;
} test;
"""
    dims = dl._dds_dimensions(dds)
    assert dims == {"time": 372, "ylat": 721, "xlon": 1440}


def test_build_socat_subset_url_uses_coordinates(monkeypatch):
    """Verify the index-math path against canned metadata/axis values."""

    calls: list[str] = []

    def fake_fetch(url, timeout=60):
        calls.append(url)
        if url.endswith(".dds"):
            return (
                "Dataset {\n"
                "  Float64 time[time = 4];\n"
                "  Float64 latitude[latitude = 3];\n"
                "  Float64 longitude[longitude = 4];\n"
                "  Grid { Float32 v[time = 4][latitude = 3][longitude = 4]; } g;\n"
                "} d;\n"
            )
        if "longitude" in url:
            return "longitude\ndegrees_east\n0.0\n90.0\n180.0\n270.0\n"
        if "latitude" in url:
            return "latitude\ndegrees_north\n-60.0\n0.0\n60.0\n"
        if "time" in url:
            return (
                "time\nUTC\n1993-01-15T00:00:00Z\n1995-01-15T00:00:00Z\n"
                "1997-01-15T00:00:00Z\n1999-01-15T00:00:00Z\n"
            )
        return ""

    monkeypatch.setattr(dl, "_fetch_text", fake_fetch)
    url = dl.build_socat_subset_url(
        "http://x/grid",
        ("v",),
        lon_range=(-100.0, -40.0),
        lat_range=(10.0, 65.0),
        year_range=(1993, 1995),
    )
    assert url.startswith("http://x/grid.nc?v[")
    assert url.count("[") == 3  # one variable, 3 index brackets per var
    assert "1993" not in url  # indices only
    assert len(calls) >= 4


def test_download_source_auth_gated(tmp_path):
    for name in ("sss", "adt", "wspd", "bathymetry"):
        with pytest.raises(RuntimeError, match="requires external access"):
            dl.download_source(name, root=tmp_path)


def test_list_month_files_regex(monkeypatch):
    """The OISST month-directory parser extracts daily .nc names."""

    def fake_fetch(url, timeout=60):
        return (
            '<a href="..">../</a>\n'
            '<a href="oisst-avhrr-v02r01.19930102.nc">..nc</a>\n'
            '<a href="oisst-avhrr-v02r01.19930101.nc">..nc</a>\n'
            '<a href="README.txt">readme</a>\n'
        )

    monkeypatch.setattr(dl, "_fetch_text", fake_fetch)
    files = dl._list_month_files("http://x/199301/")
    assert files == ["oisst-avhrr-v02r01.19930101.nc", "oisst-avhrr-v02r01.19930102.nc"]


def test_download_sst_routes_through_files(monkeypatch, tmp_path):
    """download_source('sst') no longer requires auth and writes to sst/."""
    from pathlib import Path

    calls: list[tuple[str, Path]] = []

    def fake_download(url, dest, **kw):
        calls.append((url, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"dummy")
        return dest

    def fake_list(month_url):
        m = month_url.rstrip("/").rsplit("/", 1)[-1]
        return [f"oisst-avhrr-v02r01.{m}01.nc", f"oisst-avhrr-v02r01.{m}02.nc"]

    monkeypatch.setattr(dl, "download_file", fake_download)
    monkeypatch.setattr(dl, "_list_month_files", fake_list)
    out = dl.download_source("sst", root=tmp_path, years=(1993, 1993), months=("01",))
    assert out == tmp_path / "sst"
    assert len(calls) == 2
    assert calls[0][1].parent == tmp_path / "sst"
    assert calls[0][1].name.endswith(".nc")
    assert "199301" in calls[0][0]


def test_unknown_source_rejected(tmp_path):
    with pytest.raises(ValueError):
        dl.download_source("nope", root=tmp_path)
