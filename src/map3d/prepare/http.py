"""HTTP for ``map3d prepare``: retries, throttling, cached downloads, range reads.

:class:`HttpRangeFile` is a seekable file over HTTP ``Range`` requests. Passed
to rasterio as an ``opener``, it lets GDAL do windowed reads of remote COGs
through this session's TLS settings rather than GDAL's bundled curl. That
matters on networks whose CA only Python can be told about.
"""

from __future__ import annotations

import io
import os
import ssl
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from map3d import __version__

__all__ = ["HttpRangeFile", "HttpSession"]

_RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpSession:
    """An ``httpx`` client with retries, per-host throttling and download caching."""

    def __init__(
        self,
        caBundle: str | os.PathLike[str] | None = None,
        timeout_s: float = 120.0,
        retries: int = 4,
        backoff_s: float = 2.0,
        minInterval_s: dict[str, float] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a session.

        Args:
            caBundle: PEM file of trusted roots (default: certifi).
            timeout_s: Per-request timeout.
            retries: Retries after the first attempt for transient failures.
            backoff_s: Initial backoff, doubled on each retry.
            minInterval_s: Minimum seconds between requests, per host name.
            transport: Custom transport (tests).
            sleep: Sleep function (tests).
            clock: Monotonic clock (tests).
        """
        self.sslContext: ssl.SSLContext | None = (
            ssl.create_default_context(cafile=str(caBundle)) if caBundle is not None else None
        )
        self.caBundle = None if caBundle is None else Path(caBundle)
        self._client = httpx.Client(
            timeout=timeout_s,
            follow_redirects=True,
            verify=self.sslContext if self.sslContext is not None else True,
            transport=transport,
            headers={"User-Agent": f"map3d-prepare/{__version__}"},
        )
        self._retries = retries
        self._backoff_s = backoff_s
        self._minInterval = dict(minInterval_s or {})
        self._lastRequest: dict[str, float] = {}
        self._sleep = sleep
        self._clock = clock

    def __enter__(self) -> HttpSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying client."""
        self._client.close()

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET with retries; raises for non-2xx responses.

        Args:
            url: URL.
            **kwargs: Passed to ``httpx.Client.request`` (params, headers, ...).

        Returns:
            The response.
        """
        return self._request("GET", url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> httpx.Response:
        """HEAD with retries; raises for non-2xx responses."""
        return self._request("HEAD", url, **kwargs)

    def download(self, url: str, target: str | os.PathLike[str]) -> Path:
        """Download ``url`` to ``target`` unless it is already there.

        Args:
            url: URL.
            target: Destination path (cache).

        Returns:
            The destination path.
        """
        path = Path(target)
        if path.is_file():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        response = self.get(url)
        part.write_bytes(response.content)
        os.replace(part, path)
        return path

    def rasterOpener(self, path: str, mode: str = "rb") -> HttpRangeFile:
        """rasterio ``opener``: open a URL as a seekable range-read file."""
        return HttpRangeFile(self, str(path))

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        attempt = 0
        while True:
            self._throttle(url)
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError:
                if attempt >= self._retries:
                    raise
                self._sleep(self._backoff_s * 2**attempt)
                attempt += 1
                continue
            if response.status_code in _RETRY_STATUS and attempt < self._retries:
                retryAfter = response.headers.get("retry-after", "")
                delay = float(retryAfter) if retryAfter.isdigit() else self._backoff_s * 2**attempt
                self._sleep(delay)
                attempt += 1
                continue
            response.raise_for_status()
            return response

    def _throttle(self, url: str) -> None:
        host = urlsplit(url).hostname or ""
        interval = self._minInterval.get(host)
        if interval is not None and host in self._lastRequest:
            wait = self._lastRequest[host] + interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._lastRequest[host] = self._clock()


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file backed by HTTP range requests with an LRU block cache."""

    def __init__(
        self, session: HttpSession, url: str, blockSize: int = 1 << 20, maxCachedBlocks: int = 256
    ) -> None:
        """Open a remote file.

        Args:
            session: HTTP session.
            url: File URL.
            blockSize: Bytes per range request.
            maxCachedBlocks: Blocks kept in memory.

        Raises:
            FileNotFoundError: If ``url`` is not http(s) or the server answers 404
                (GDAL probes sidecar files; rasterio probes a dummy path).
        """
        super().__init__()
        if urlsplit(url).scheme not in ("http", "https"):
            raise FileNotFoundError(url)
        self._session = session
        self._url = url
        self._blockSize = blockSize
        self._maxCached = maxCachedBlocks
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._pos = 0
        try:
            response = session.head(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise FileNotFoundError(url) from None
            raise
        self.size = int(response.headers["content-length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def read(self, size: int | None = -1) -> bytes:
        remaining = self.size - self._pos
        n = remaining if size is None or size < 0 else min(size, remaining)
        out = bytearray()
        while n > 0:
            index, offset = divmod(self._pos, self._blockSize)
            chunk = self._block(index)[offset : offset + n]
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def _block(self, index: int) -> bytes:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        start = index * self._blockSize
        end = min(self.size, start + self._blockSize) - 1
        data = self._session.get(self._url, headers={"Range": f"bytes={start}-{end}"}).content
        self._cache[index] = data
        if len(self._cache) > self._maxCached:
            self._cache.popitem(last=False)
        return data
