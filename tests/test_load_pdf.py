"""hybrid 로더(표 페이지만 마크다운 교체) 단위 테스트.

실제 PDF·pymupdf 없이 동작을 검증하기 위해, 페이지 로드 함수만 가짜로 바꿔치기한다.
(load_pdf 모듈은 langchain 의존이 있어 import 자체는 필요하나, 테스트 대상 로직은 순수 파이썬)
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

load_pdf = pytest.importorskip("load_pdf")


class FakeDoc:
    def __init__(self, page, text):
        self.metadata = {"page": page}
        self.page_content = text


@pytest.fixture
def fake_pages(monkeypatch):
    """pypdf 3페이지 / markdown 3페이지를 가짜로 제공한다."""
    pypdf_docs = [FakeDoc(0, "본문0-pypdf"), FakeDoc(1, "표1-pypdf"), FakeDoc(2, "본문2-pypdf")]
    md_docs = [FakeDoc(0, "본문0-md"), FakeDoc(1, "| 표 | 1 |\n|---|---|"), FakeDoc(2, "본문2-md")]

    def fake_load_documents(pdf_path=None, loader=None):
        return pypdf_docs

    monkeypatch.setattr(load_pdf, "load_documents", fake_load_documents)
    monkeypatch.setattr(load_pdf, "load_documents_markdown", lambda pdf_path=None: md_docs)
    return pypdf_docs


def test_hybrid_replaces_only_table_pages(monkeypatch, fake_pages):
    """표로 검출된 페이지만 마크다운으로 바뀌고 나머지는 pypdf 원문을 유지한다."""
    monkeypatch.setattr(load_pdf, "detect_table_pages", lambda pdf_path=None, min_rows=3: {1})

    docs = load_pdf.load_documents_hybrid()

    assert docs[0].page_content == "본문0-pypdf"          # 표 아님 → 그대로
    assert docs[1].page_content.startswith("| 표 |")      # 표 → 마크다운으로 교체
    assert docs[2].page_content == "본문2-pypdf"          # 표 아님 → 그대로


def test_hybrid_tags_loader_in_metadata(monkeypatch, fake_pages):
    """어느 페이지가 교체됐는지 metadata['loader']로 추적할 수 있다."""
    monkeypatch.setattr(load_pdf, "detect_table_pages", lambda pdf_path=None, min_rows=3: {1})

    docs = load_pdf.load_documents_hybrid()

    assert [d.metadata["loader"] for d in docs] == ["pypdf", "markdown", "pypdf"]


def test_hybrid_keeps_pypdf_when_markdown_page_is_empty(monkeypatch, fake_pages):
    """마크다운 변환 결과가 비면 교체하지 않고 pypdf 원문을 유지한다(빈 문맥 방지)."""
    monkeypatch.setattr(load_pdf, "detect_table_pages", lambda pdf_path=None, min_rows=3: {1})
    monkeypatch.setattr(load_pdf, "load_documents_markdown",
                        lambda pdf_path=None: [FakeDoc(0, "본문0-md"), FakeDoc(1, "   "), FakeDoc(2, "본문2-md")])

    docs = load_pdf.load_documents_hybrid()

    assert docs[1].page_content == "표1-pypdf"
    assert docs[1].metadata["loader"] == "pypdf"


def test_hybrid_without_tables_is_identical_to_pypdf(monkeypatch, fake_pages):
    """표가 하나도 검출되지 않으면 pypdf 로더와 완전히 동일한 결과여야 한다."""
    monkeypatch.setattr(load_pdf, "detect_table_pages", lambda pdf_path=None, min_rows=3: set())

    docs = load_pdf.load_documents_hybrid()

    assert [d.page_content for d in docs] == ["본문0-pypdf", "표1-pypdf", "본문2-pypdf"]


def test_loader_choices_contains_hybrid():
    assert load_pdf.LOADER_CHOICES == ["pypdf", "markdown", "hybrid"]
