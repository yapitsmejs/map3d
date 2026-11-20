from __future__ import annotations

import map3d


def testVersionIsSet():
    assert isinstance(map3d.__version__, str)
    assert map3d.__version__
