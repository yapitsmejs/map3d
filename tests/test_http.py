"""Tests for ``map3d.prepare.http`` and ``map3d.prepare.certs`` (no live network)."""

from __future__ import annotations

import ssl
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from _packSupport import mockSession
from _support import rangeServer
from rasterio.windows import Window

from map3d.prepare import certs
from map3d.prepare.http import HttpRangeFile, HttpSession


def testRangeFile_readsSeeksAndCachesBlocks():
    data = bytes(range(256)) * 40  # 10240 bytes
    log: list[httpx.Request] = []
    with mockSession(rangeServer({"https://x/f.bin": data}, log)) as session:
        f = HttpRangeFile(session, "https://x/f.bin", blockSize=1000)
        assert f.size == len(data)
        f.seek(995)
        assert f.read(10) == data[995:1005]  # spans two blocks
        assert f.tell() == 1005
        f.seek(-5, 2)
        assert f.read() == data[-5:]
        f.seek(990)
        f.read(20)
        gets = [r for r in log if r.method == "GET"]
        assert len(gets) == 3  # blocks 0, 1 and 10; re-reads hit the cache
        buf = bytearray(4)
        f.seek(0)
        assert f.readinto(buf) == 4 and bytes(buf) == data[:4]
        f.seek(2, 1)
        assert f.tell() == 6
        assert f.readable() and f.seekable()


def testRangeFile_evictsLeastRecentlyUsedBlocks():
    data = bytes(5000)
    with mockSession(rangeServer({"https://x/f": data})) as session:
        f = HttpRangeFile(session, "https://x/f", blockSize=1000, maxCachedBlocks=2)
        for pos in (0, 1000, 2000):
            f.seek(pos)
            f.read(1)
        assert len(f._cache) == 2


def testRangeFile_readPastEndReturnsEmpty():
    with mockSession(rangeServer({"https://x/f": b"abc"})) as session:
        f = HttpRangeFile(session, "https://x/f")
        f.seek(10)
        assert f.read(5) == b""


def testRasterOpener_readsRemoteGeoTiffWindow(tmp_path: Path, writeRaster):
    values = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = writeRaster(tmp_path / "r.tif", (103.0, 1.0, 104.0, 2.0), values=values, shape=(64, 64))
    with mockSession(rangeServer({"https://x/r.tif": path.read_bytes()})) as session:
        with rasterio.open("https://x/r.tif", opener=session.rasterOpener) as src:
            np.testing.assert_array_equal(
                src.read(1, window=Window(10, 20, 5, 3)), values[20:23, 10:15]
            )


def testGet_retriesTransientFailuresThenSucceeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        if calls["n"] == 2:
            return httpx.Response(503)
        if calls["n"] == 3:
            return httpx.Response(429, headers={"retry-after": "1"})
        return httpx.Response(200, text="ok")

    slept: list[float] = []
    with mockSession(handler, sleep=slept.append, retries=4, backoff_s=0.5) as session:
        assert session.get("https://x/a").text == "ok"
    assert calls["n"] == 4
    assert slept[-1] == 1.0  # honoured Retry-After


def testGet_givesUpAfterRetries():
    with mockSession(lambda r: httpx.Response(500), retries=2) as session:
        with pytest.raises(httpx.HTTPStatusError):
            session.get("https://x/a")


def testGet_doesNotRetryClientErrors():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    with mockSession(handler) as session:
        with pytest.raises(httpx.HTTPStatusError):
            session.get("https://x/missing")
    assert calls["n"] == 1


def testGet_transportErrorRaisedAfterRetries():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with mockSession(handler, retries=1) as session:
        with pytest.raises(httpx.ReadTimeout):
            session.get("https://x/a")


def testThrottle_enforcesMinimumIntervalPerHost():
    clock = {"t": 100.0}
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        clock["t"] += s

    with mockSession(
        lambda r: httpx.Response(200),
        sleep=sleep,
        clock=lambda: clock["t"],
        minInterval_s={"api.example": 5.0},
    ) as session:
        session.get("https://api.example/a")
        session.get("https://api.example/b")
        session.get("https://other.example/c")
    assert slept == [5.0]


def testDownload_streamsToFileAndReusesCache(tmp_path: Path):
    log: list[httpx.Request] = []
    with mockSession(rangeServer({"https://x/big.bin": b"payload"}, log)) as session:
        target = tmp_path / "cache" / "big.bin"
        assert session.download("https://x/big.bin", target) == target
        assert target.read_bytes() == b"payload"
        session.download("https://x/big.bin", target)
    assert len(log) == 1
    assert not list(target.parent.glob("*.part"))


def testHead_returnsResponse():
    with mockSession(rangeServer({"https://x/f": b"12345"})) as session:
        assert session.head("https://x/f").headers["content-length"] == "5"


def testSession_caBundleBuildsVerifyingContext(tmp_path: Path):
    bundle = certs.exportCaBundle(tmp_path / "ca.pem")
    session = HttpSession(caBundle=bundle)
    try:
        assert isinstance(session.sslContext, ssl.SSLContext)
        assert session.sslContext.verify_mode == ssl.CERT_REQUIRED
    finally:
        session.close()


def testExportCaBundle_combinesCertifiAndSystemCerts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import certifi

    text = Path(certifi.where()).read_text(encoding="ascii")
    begin = text.index("-----BEGIN CERTIFICATE-----")
    end = text.index("-----END CERTIFICATE-----") + len("-----END CERTIFICATE-----")
    fake = ssl.PEM_cert_to_DER_cert(text[begin:end] + "\n")
    monkeypatch.setattr(certs, "_systemCertificatesDer", lambda: [fake, fake])
    path = certs.exportCaBundle(tmp_path / "sub" / "ca.pem")
    text = path.read_text(encoding="ascii")
    certifiCount = Path(certifi.where()).read_text(encoding="ascii").count("BEGIN CERTIFICATE")
    assert text.count("BEGIN CERTIFICATE") == certifiCount + 2


def testSystemCertificates_nonWindowsIsEmpty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delattr(ssl, "enum_certificates", raising=False)
    assert certs._systemCertificatesDer() == []


def testSystemCertificates_windowsEnumeratesX509Only(monkeypatch: pytest.MonkeyPatch):
    def enum(store: str):
        return [(b"der-" + store.encode(), "x509_asn", True), (b"crl", "pkcs_7_asn", True)]

    monkeypatch.setattr(ssl, "enum_certificates", enum, raising=False)
    assert certs._systemCertificatesDer() == [b"der-ROOT", b"der-CA"]


def testRangeFile_nonHttpPathIsNotFound():
    with mockSession(rangeServer({})) as session:
        with pytest.raises(FileNotFoundError):
            session.rasterOpener("test/test")


def testRangeFile_404IsNotFoundOtherErrorsPropagate():
    with mockSession(rangeServer({})) as session:
        with pytest.raises(FileNotFoundError):
            HttpRangeFile(session, "https://x/missing.tif.aux.xml")
    with mockSession(lambda r: httpx.Response(403)) as session:
        with pytest.raises(httpx.HTTPStatusError):
            HttpRangeFile(session, "https://x/forbidden.tif")
