"""Shared test helpers (importable from tests: ``from _support import ...``)."""

from __future__ import annotations

import httpx


def rangeServer(files: dict[str, bytes], log: list[httpx.Request] | None = None):
    """MockTransport handler serving HEAD and byte-range GETs for ``files``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if log is not None:
            log.append(request)
        body = files.get(str(request.url))
        if body is None:
            return httpx.Response(404)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(body))})
        rng = request.headers.get("range")
        if rng:
            start, end = (int(v) for v in rng.removeprefix("bytes=").split("-"))
            part = body[start : end + 1]
            return httpx.Response(
                206,
                content=part,
                headers={"content-range": f"bytes {start}-{start + len(part) - 1}/{len(body)}"},
            )
        return httpx.Response(200, content=body)

    return handler
