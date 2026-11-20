"""Design-point benchmark: build (and optionally write) a large scene, report time and memory.

Usage::

    uv run python scripts/benchmark.py --pack marina.pack --lat 1.285 --lon 103.86 \\
        --spacing 0.5 --swath 5000 --out bench-out

Reports wall time per stage and peak memory. On Windows it reports both the
private commit (RAM the process really owns) and the working set, which also
counts reclaimable file-backed pages of memory-mapped layers. On other systems
it reports the max RSS.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path


def _peakMemory_GB() -> dict[str, float]:
    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        if not kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return {
            "peakPrivate_GB": counters.PeakPagefileUsage / 1e9,
            "peakWorkingSet_GB": counters.PeakWorkingSetSize / 1e9,
        }
    import resource

    return {"peakRss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--swath", type=float, default=5000.0)
    parser.add_argument("--decks", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=None, help="Also write npz, GeoTIFF and 3D Tiles here."
    )
    args = parser.parse_args()

    from map3d import SceneRequest, buildScene

    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    warm = SceneRequest((args.lat, args.lon), 20.0, identity, 400.0, 400.0)
    buildScene(warm, args.pack)  # JIT/cache warm-up, excluded from timings
    request = SceneRequest(
        (args.lat, args.lon),
        args.spacing,
        identity,
        args.swath,
        args.swath,
        maxDeckLevels=args.decks,
    )
    workDir = None if args.out is None else args.out / "_layers"
    t0 = time.perf_counter()
    scene = buildScene(request, args.pack, workers=args.workers, workDir=workDir)
    t1 = time.perf_counter()
    nodes = scene.geometry.nRows * scene.geometry.nCols
    print(f"nodes: {nodes:,}  storage: {scene.metadata['storage']}")
    print(f"build: {t1 - t0:.1f} s")
    if args.out is not None:
        scene.writeNpz(args.out / "scene")
        t2 = time.perf_counter()
        print(f"write npz: {t2 - t1:.1f} s")
        scene.writeGeoTiff(args.out / "geotiff")
        t3 = time.perf_counter()
        print(f"write geotiff: {t3 - t2:.1f} s")
        scene.write3dTiles(args.out / "tiles")
        t4 = time.perf_counter()
        print(f"write 3D Tiles: {t4 - t3:.1f} s")
        print(f"total: {t4 - t0:.1f} s")
    scene.close()
    for name, value in _peakMemory_GB().items():
        print(f"{name}: {value:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
