"""
[담당: 이수민] 문서 로드 및 청킹.

국세청 「연말정산 신고안내」 PDF를 로드하고 청크로 분할한다.
- PyPDFLoader로 페이지 단위 로드 (page 메타데이터 자동 부여)
- RecursiveCharacterTextSplitter로 분할, chunk_id 메타데이터 추가

실행 예:
    python src/load_pdf.py                      # 기본값 (chunk_size=1000, overlap=200)
    python src/load_pdf.py --chunk-size 500 --overlap 100
    python src/load_pdf.py --loader markdown     # 마크다운 변환(pymupdf4llm, 표·구조 보존)
    python src/load_pdf.py --loader hybrid       # 표 페이지만 마크다운, 나머지는 pypdf
    python src/load_pdf.py --show-table-pages    # hybrid가 교체할 표 페이지 목록 확인
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

# hybrid 로더에서 "표"로 인정할 최소 행 수 (머리말·목차의 1~2행 오검출 제외)
MIN_TABLE_ROWS = 3


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


def detect_table_pages(pdf_path=config.PDF_PATH, min_rows=MIN_TABLE_ROWS):
    """표가 있는 페이지(0-indexed)를 자동 검출한다.

    PyMuPDF의 find_tables()로 페이지마다 표 유무를 판정한다. 특정 문항의 정답
    페이지를 손으로 지정하지 않는 **일반 규칙**이어야 과적합을 피할 수 있어서,
    "표가 검출된 페이지"라는 문서 구조 기준만 사용한다.

    min_rows: 이 행 수 미만의 표는 무시한다. 머리말·목차 등에서 잡히는 1~2행짜리
              오검출을 걸러, 실제 조건-금액 대응이 있는 표만 남기기 위함.
    """
    import pymupdf

    table_pages = set()
    with pymupdf.open(str(pdf_path)) as doc:
        for page_index, page in enumerate(doc):
            for table in page.find_tables().tables:
                if table.row_count >= min_rows:
                    table_pages.add(page_index)
                    break
    return table_pages


def load_documents_hybrid(pdf_path=config.PDF_PATH, min_rows=MIN_TABLE_ROWS):
    """표가 있는 페이지만 마크다운으로, 나머지 페이지는 pypdf 텍스트로 로드한다.

    ※ 배경 (evalset26_generalization.md → 검증):
      - pypdf 전용: Q18(장기주택저당차입금 한도표) 실패. 정답 p.133 대신 이웃
        페이지 [234,136,132]만 검색됨. 표의 "상환기간 15년 이상" 같은 상위 조건이
        아래 행들에 반복되지 않아 조건-금액 대응이 끊기고, 깔끔하게 재정리된
        목록 부분은 청크 경계에서 주제어("장기주택저당차입금")와 분리된다.
      - markdown 전용: Q18·Q14는 해결되지만 Q6·Q10이 새로 깨지고 MRR이
        0.826 → 0.638로 하락. 표가 아닌 본문까지 마크다운 서식이 섞이면서
        다른 문항의 임베딩 품질을 깎아먹기 때문.
      - 따라서 **표 페이지만** 마크다운으로 바꾸고 나머지는 그대로 둔다.
        chunk_size/overlap 등 확정된 전역 파라미터는 건드리지 않으므로
        기존 실험 결과가 그대로 유효하다.
    """
    documents = load_documents(pdf_path, loader="pypdf")
    table_pages = detect_table_pages(pdf_path, min_rows)

    md_documents = {d.metadata["page"]: d for d in load_documents_markdown(pdf_path)}

    replaced = 0
    for document in documents:
        page_index = int(document.metadata["page"])
        md_document = md_documents.get(page_index)
        if page_index in table_pages and md_document and md_document.page_content.strip():
            document.page_content = md_document.page_content
            document.metadata["loader"] = "markdown"     # 어느 페이지가 교체됐는지 추적용
            replaced += 1
        else:
            document.metadata["loader"] = "pypdf"

    print(f"[load_pdf] (hybrid) 표 검출 페이지 {len(table_pages)}개 중 {replaced}개를 마크다운으로 교체")
    return documents


def load_documents(pdf_path=config.PDF_PATH, loader=None):
    """PDF를 페이지 단위 Document 리스트로 로드한다.

    loader="pypdf"(기본, 일반 텍스트) | "markdown"(pymupdf4llm, 전체 변환)
         | "hybrid"(표 페이지만 마크다운). None이면 config.PDF_LOADER를 따른다.
    """

    loader = loader or config.PDF_LOADER
    if loader == "markdown":
        return load_documents_markdown(pdf_path)
    if loader == "hybrid":
        return load_documents_hybrid(pdf_path)

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


LOADER_CHOICES = ["pypdf", "markdown", "hybrid"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF 로드 및 청킹 테스트")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--loader", choices=LOADER_CHOICES, default=config.PDF_LOADER)
    parser.add_argument("--show-table-pages", action="store_true",
                        help="hybrid가 마크다운으로 교체할 표 페이지 목록만 출력 (과적합 점검용)")
    args = parser.parse_args()

    if args.show_table_pages:
        pages = sorted(p + 1 for p in detect_table_pages())   # 물리 페이지(1-indexed)로 출력
        print(f"[load_pdf] 표 검출 페이지 {len(pages)}개 (물리 페이지 기준):")
        print(pages)
        raise SystemExit(0)

    chunks = get_chunks(args.chunk_size, args.overlap, loader=args.loader)

    # 샘플 청크 확인
    for chunk in chunks[:3]:
        print("-" * 60)
        print(f"페이지: {chunk.metadata.get('page')}, chunk_id: {chunk.metadata.get('chunk_id')}")
        print(chunk.page_content[:200])
