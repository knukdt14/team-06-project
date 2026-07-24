"""
[담당: 이수민] 문서 로드 및 청킹.

국세청 「연말정산 신고안내」 PDF를 로드하고 청크로 분할한다.
- PyPDFLoader로 페이지 단위 로드 (page 메타데이터 자동 부여)
- RecursiveCharacterTextSplitter로 분할, chunk_id 메타데이터 추가

실행 예:
    python src/load_pdf.py                      # 기본값 (chunk_size=1000, overlap=200)
    python src/load_pdf.py --chunk-size 500 --overlap 100

"""
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_documents(pdf_path=config.PDF_PATH):
    """PDF를 페이지 단위 Document 리스트로 로드한다."""

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"[load_pdf] {pdf_path.name}: {len(documents)} 페이지 로드 완료")
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


def get_chunks(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """로드 + 분할을 한 번에 수행 (build_vectorstore에서 사용)."""
    return split_documents(load_documents(), chunk_size, overlap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF 로드 및 청킹 테스트")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    chunks = get_chunks(args.chunk_size, args.overlap)

    # 샘플 청크 확인
    for chunk in chunks[:3]:
        print("-" * 60)
        print(f"페이지: {chunk.metadata.get('page')}, chunk_id: {chunk.metadata.get('chunk_id')}")
        print(chunk.page_content[:200])
