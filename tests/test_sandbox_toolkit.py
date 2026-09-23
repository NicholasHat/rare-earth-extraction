"""Tests for the digitisation toolkit shipped into the code-execution sandbox
(extraction/sandbox_toolkit.py): the bundle must load where pdfplumber and
scikit-image are absent, and the guide must describe functions that exist."""
import importlib
import io
import re
import sys
import zipfile

import numpy as np
import pytest

from extraction import sandbox_toolkit


def test_bundle_is_deterministic_and_holds_the_package():
    a, b = sandbox_toolkit.bundle(), sandbox_toolkit.bundle()
    assert a == b and sandbox_toolkit.sha256() == sandbox_toolkit.sha256()
    names = zipfile.ZipFile(io.BytesIO(a)).namelist()
    assert "curve_extractor/__init__.py" in names
    assert {"curve_extractor/raster.py", "curve_extractor/calibrate.py"} <= set(names)
    assert all(n.startswith("curve_extractor/") and n.endswith(".py") for n in names)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """The bundle unpacked into a bare directory, imported as the sandbox would
    import it: top-level `curve_extractor`, with pdfplumber and scikit-image
    unavailable (their absence is the sandbox's documented state)."""
    zipfile.ZipFile(io.BytesIO(sandbox_toolkit.bundle())).extractall(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("pdfplumber", "skimage", "skimage.feature"):
        monkeypatch.setitem(sys.modules, name, None)   # None => ImportError on import
    stale = [m for m in sys.modules if m == "curve_extractor" or m.startswith("curve_extractor.")]
    for m in stale:
        monkeypatch.delitem(sys.modules, m)
    pkg = importlib.import_module("curve_extractor")
    yield pkg
    for m in [m for m in sys.modules if m == "curve_extractor" or m.startswith("curve_extractor.")]:
        sys.modules.pop(m, None)


def test_bundle_imports_and_digitises_without_pdfplumber_or_scikit_image(sandbox):
    raster = importlib.import_module("curve_extractor.raster")
    calibrate = importlib.import_module("curve_extractor.calibrate")
    img = np.full((300, 400), 255, dtype=np.uint8)
    img[20:22, 40:381] = 0; img[259:261, 40:381] = 0; img[20:261, 40:42] = 0; img[20:261, 379:381] = 0
    for cx in range(80, 340, 40):
        img[94:106, cx - 6:cx + 6] = 0
    records, warnings = raster.detect_markers_in_image(img)
    assert len(records) == 7 and {r.group_key for r in records} == {"filled_square"}
    assert any("scikit-image" in w for w in warnings)
    assert raster.find_frame(img) == (40, 20, 380, 260)
    cal = calibrate.fit_axis("x", [80, 160, 240, 320], [1, 2, 3, 4])
    assert cal.pixel_to_data(200) == pytest.approx(2.5)


def test_guide_names_only_functions_that_exist():
    from extraction.curve_extractor import calibrate, raster
    guide = sandbox_toolkit.guide()
    named = set(re.findall(r"`(raster|calibrate)\.(\w+)\(", guide))
    assert named, "guide should document toolkit calls"
    for module, fn in named:
        assert callable(getattr({"raster": raster, "calibrate": calibrate}[module], fn)), f"{module}.{fn}"
    assert sandbox_toolkit.FILENAME in guide
    assert "pdfplumber" in guide and "fitz" in guide     # the environment facts the model used to probe for
