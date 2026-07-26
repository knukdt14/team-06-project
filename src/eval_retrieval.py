"""[담당: 이수민] 검색 품질 채점 (LLM/API 불필요).

references.csv의 정답 page를 이용해, 질문마다 정답 페이지가 검색된 top-k 청크의
페이지 목록에 포함되는지 채점한다. Hit@k와 MRR을 계산한다.

페이지 정합: chunk.metadata['page']는 0-indexed(PyPDF), references.csv의 page는
1-indexed 물리 페이지이므로, chunk page에 +1을 하여 맞춘다.
"""


def retrieved_physical_pages(docs):
    """검색된 청크들의 물리 페이지(1-indexed) 목록. page 메타데이터 없으면 -1."""
    pages = []
    for d in docs:
        p = d.metadata.get("page")
        pages.append(p + 1 if p is not None else -1)
    return pages


def page_hit(retrieved_pages, gold_page):
    """정답 페이지가 검색된 페이지 목록에 있으면 True."""
    return gold_page in retrieved_pages


def reciprocal_rank(retrieved_pages, gold_page):
    """정답 페이지가 처음 등장한 순위의 역수(1/rank). 없으면 0.0."""
    for i, p in enumerate(retrieved_pages, start=1):
        if p == gold_page:
            return 1.0 / i
    return 0.0
