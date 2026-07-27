"""
[담당: 이수민] 문서 로드 및 청킹.

국세청 「연말정산 신고안내」 PDF를 로드하고 청크로 분할한다.
- PyPDFLoader로 페이지 단위 로드 (page 메타데이터 자동 부여)
- RecursiveCharacterTextSplitter로 분할, chunk_id 메타데이터 추가

실행 예:
    python src/load_pdf.py                      # 기본값 (chunk_size=1000, overlap=200)
    python src/load_pdf.py --chunk-size 500 --overlap 100
    python src/load_pdf.py --loader markdown     # 마크다운 변환(pymupdf4llm, 표·구조 보존)
"""
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


REPEATED_HEADER_LINES = (
    "원천징수의무자를 위한",
    "2025년 연말정산 신고안내",
)
PRINTED_PAGE_OFFSET = 18


def remove_repeated_header(text, page_index):
    """본문 페이지 상단의 반복 머리말과 인쇄 페이지번호를 제거한다."""

    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    if (
        len(lines) >= 2
        and lines[0].strip() == REPEATED_HEADER_LINES[0]
        and lines[1].strip() == REPEATED_HEADER_LINES[1]
    ):
        del lines[:2]

    physical_page = page_index + 1
    expected_printed_page = physical_page - PRINTED_PAGE_OFFSET
    if (
        expected_printed_page > 0
        and lines
        and lines[0].strip() == str(expected_printed_page)
    ):
        del lines[0]

    return "\n".join(lines).strip()


def _strip_markdown_prefix(line):
    """마크다운 장식(#, *, -, >)과 공백을 벗겨 머리말 비교용 텍스트만 남긴다."""
    return line.strip().lstrip("#*->").strip()


def remove_repeated_header_md(text, page_index):
    """마크다운 페이지 상단의 반복 머리말·인쇄 페이지번호를 제거한다.

    마크다운 변환 시 머리말이 '원천징수의무자를 위한 2025년 연말정산 신고안내'처럼
    한 줄로 합쳐져 나오는 경우가 있어, 일반 텍스트용(remove_repeated_header)과 별도로 처리한다.
    """
    combined = " ".join(REPEATED_HEADER_LINES)
    expected_printed_page = (page_index + 1) - PRINTED_PAGE_OFFSET

    lines = text.splitlines()
    while lines:                                   # 상단의 빈 줄·머리말·페이지번호를 반복 제거
        head = _strip_markdown_prefix(lines[0])
        if not head:
            lines.pop(0)
        elif head.startswith(combined) or head in REPEATED_HEADER_LINES:
            lines.pop(0)
        elif expected_printed_page > 0 and head == str(expected_printed_page):
            lines.pop(0)
        else:
            break                                  # 실제 본문 시작 → 중단(과잉 삭제 방지)
    return "\n".join(lines).strip()


def load_documents_markdown(pdf_path=config.PDF_PATH):
    """PDF를 pymupdf4llm로 마크다운 변환해 페이지 단위 Document로 로드한다.

    - page_chunks=True로 페이지별 마크다운을 얻고, 리스트 인덱스를 page(0-indexed)로 부여한다.
      → PyPDFLoader와 동일한 page 메타데이터 규칙이라 평가(물리 페이지 채점)와 호환된다.
    - 표·목록 등 문서 구조가 마크다운(| |, #, -)으로 보존된다.
    """
    import pymupdf4llm
    from langchain_core.documents import Document

    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    documents = []
    cleaned_pages = 0
    for i, chunk in enumerate(page_chunks):
        # pymupdf4llm 메타의 page_number(1-indexed)를 0-indexed로 변환해 page로 사용.
        # (없으면 순회 인덱스로 대체) → PyPDFLoader의 page 규칙과 동일하게 맞춘다.
        page_number = chunk.get("metadata", {}).get("page_number")
        page_index = (page_number - 1) if page_number else i
        original = chunk["text"]
        cleaned = remove_repeated_header_md(original, page_index)
        if cleaned != original.strip():
            cleaned_pages += 1
        documents.append(Document(page_content=cleaned, metadata={"page": page_index}))

    print(f"[load_pdf] (markdown) {pdf_path.name}: {len(documents)} 페이지 로드 완료")
    print(f"[load_pdf] (markdown) 반복 머리말·페이지번호 제거: {cleaned_pages} 페이지")
    return documents


def load_documents(pdf_path=config.PDF_PATH, loader=None):
    """PDF를 페이지 단위 Document 리스트로 로드한다.

    loader="pypdf"(기본, 일반 텍스트) | "markdown"(pymupdf4llm). None이면 config.PDF_LOADER를 따른다.
    """

    loader = loader or config.PDF_LOADER
    if loader == "markdown":
        return load_documents_markdown(pdf_path)

    pdf_loader = PyPDFLoader(str(pdf_path))
    documents = pdf_loader.load()
    cleaned_pages = 0
    for document in documents:
        original = document.page_content
        page_index = int(document.metadata["page"])
        document.page_content = remove_repeated_header(original, page_index)
        if document.page_content != original.strip():
            cleaned_pages += 1

    print(f"[load_pdf] {pdf_path.name}: {len(documents)} 페이지 로드 완료")
    print(f"[load_pdf] 반복 머리말·페이지번호 제거: {cleaned_pages} 페이지")
    return documents


def split_documents(documents, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """문서를 청크로 분할하고 chunk_id 메타데이터를 부여한다."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    print(f"[load_pdf] chunk_size={chunk_size}, overlap={overlap} → {len(chunks)} 청크 생성")
    return chunks


def get_chunks(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP, loader=None):
    """로드 + 분할을 한 번에 수행 (build_vectorstore에서 사용). loader로 로더 방식 선택."""

    return split_documents(load_documents(loader=loader), chunk_size, overlap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF 로드 및 청킹 테스트")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--loader", choices=["pypdf", "markdown"], default=config.PDF_LOADER)
    args = parser.parse_args()

    chunks = get_chunks(args.chunk_size, args.overlap, loader=args.loader)

    # 샘플 청크 확인
    for chunk in chunks[:3]:
        print("-" * 60)
        print(f"페이지: {chunk.metadata.get('page')}, chunk_id: {chunk.metadata.get('chunk_id')}")
        print(chunk.page_content[:200])
