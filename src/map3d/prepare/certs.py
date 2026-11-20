"""CA bundle export for networks with TLS interception.

Some organisations re-sign TLS traffic with their own root CA, installed in the
operating system's trust store. Python (certifi), GDAL's curl and Arrow's S3
client don't read that store. :func:`exportCaBundle` writes certifi's roots plus
the OS roots to one PEM file that all of them can be pointed at. Verification
stays fully enabled.
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path

__all__ = ["exportCaBundle"]


def _systemCertificatesDer() -> list[bytes]:
    """DER certificates from the Windows ROOT and CA stores (empty elsewhere)."""
    enum = getattr(ssl, "enum_certificates", None)
    if enum is None:
        return []
    ders: list[bytes] = []
    for store in ("ROOT", "CA"):
        for cert, encoding, _trust in enum(store):
            if encoding == "x509_asn":
                ders.append(bytes(cert))
    return ders


def exportCaBundle(path: str | os.PathLike[str]) -> Path:
    """Write certifi's CA roots plus the operating system's roots to a PEM file.

    On non-Windows systems only certifi is included; OpenSSL there usually
    reads the system store already.

    Args:
        path: Output PEM path (parent directories are created).

    Returns:
        The written path.
    """
    import certifi

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pems = [Path(certifi.where()).read_text(encoding="ascii")]
    pems += [ssl.DER_cert_to_PEM_cert(der) for der in _systemCertificatesDer()]
    out.write_text("\n".join(pems), encoding="ascii")
    return out
