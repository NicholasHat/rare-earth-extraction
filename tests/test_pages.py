"""curve_extractor.pages renders in pdfplumber's coordinates with any backend."""
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, RectangleObject

from extraction.curve_extractor import pages


@pytest.fixture
def trimmed_pdf(tmp_path):
    """A 842x1191 page whose crop box trims (42, 351) pt off its top-left
    (pdfplumber space), with a black box at x 140-400, top 500-680."""
    writer = PdfWriter()
    page = writer.add_blank_page(842, 1191)
    stream = DecodedStreamObject()
    stream.set_data(b"0 g 140 511 260 180 re f")           # PDF space: bottom-left origin
    page[NameObject("/Contents")] = writer._add_object(stream)
    page.cropbox = RectangleObject([42, 47, 637, 840])
    path = tmp_path / "trimmed.pdf"
    with open(path, "wb") as f:
        writer.write(f)
    return str(path)


@pytest.mark.parametrize("backend", ["_pdfium", "_plumber"])
def test_render_addresses_the_box_in_pdfplumber_space(trimmed_pdf, backend, monkeypatch):
    others = {"_pdfium", "_mupdf", "_plumber"} - {backend}
    for name in others:   # force this backend
        monkeypatch.setattr(pages, name, lambda *a, **k: (_ for _ in ()).throw(ImportError("off")))
    inside = pages.render(trimmed_pdf, 0, (200, 550, 300, 650), dpi=72)
    outside = pages.render(trimmed_pdf, 0, (450, 550, 550, 650), dpi=72)
    assert inside.size and inside.mean() < 10
    assert outside.size and outside.mean() > 245


def test_no_backend_is_a_clear_error(trimmed_pdf, monkeypatch):
    for name in ("_pdfium", "_mupdf", "_plumber"):
        monkeypatch.setattr(pages, name, lambda *a, **k: (_ for _ in ()).throw(ImportError(name)))
    with pytest.raises(ImportError, match="no PDF renderer"):
        pages.render(trimmed_pdf, 0)


def test_prepass_render_region_uses_the_same_space(trimmed_pdf):
    import pdfplumber
    from extraction.curve_extractor import raster
    with pdfplumber.open(trimmed_pdf) as pdf:
        assert raster.render_region(pdf.pages[0], (200, 550, 300, 650), dpi=72).mean() < 10
