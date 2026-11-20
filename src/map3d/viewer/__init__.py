"""Offline 3D Tiles viewer: static files (three.js, MIT) plus a localhost server.

:func:`installViewer` copies the viewer next to a ``tileset.json``, and
:func:`serve` serves that directory over HTTP on localhost. Browsers won't
``fetch`` from ``file://``, and ES modules need a correct JavaScript MIME type,
so explicit MIME types are set rather than trusting the platform registry.
"""

from __future__ import annotations

import functools
import http.server
import os
import shutil
import threading
import webbrowser
from pathlib import Path
from typing import Any

__all__ = ["STATIC_DIR", "installViewer", "serve"]

STATIC_DIR = Path(__file__).parent / "static"

_MIME = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".glb": "model/gltf-binary",
    ".css": "text/css",
    "": "application/octet-stream",
}


def installViewer(outDir: str | os.PathLike[str]) -> Path:
    """Copy the viewer's static files into ``outDir`` (next to ``tileset.json``).

    Args:
        outDir: Tileset directory.

    Returns:
        Path of the installed ``index.html``.
    """
    target = Path(outDir)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(STATIC_DIR, target, dirs_exist_ok=True)
    return target / "index.html"


class _Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = _MIME

    def log_message(self, format: str, *args: Any) -> None:  # quiet by default
        pass


def serve(
    directory: str | os.PathLike[str], port: int = 8000, openBrowser: bool = True
) -> http.server.ThreadingHTTPServer:
    """Serve ``directory`` on ``http://127.0.0.1:<port>/`` in a background thread.

    Args:
        directory: Directory holding ``index.html`` and ``tileset.json``.
        port: TCP port (0 picks a free one).
        openBrowser: Open the default browser at the viewer URL.

    Returns:
        The running server; call ``shutdown()`` to stop it.

    Raises:
        FileNotFoundError: If the directory has no ``tileset.json``.
    """
    root = Path(directory)
    if not (root / "tileset.json").is_file():
        raise FileNotFoundError(f"no tileset.json in {root}")
    if not (root / "index.html").is_file():
        installViewer(root)
    handler = functools.partial(_Handler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if openBrowser:
        webbrowser.open(f"http://127.0.0.1:{server.server_address[1]}/")
    return server
