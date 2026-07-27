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


def parse_gold_pages(page_field):
    """references.csv의 page 필드를 정답 페이지 목록으로 파싱한다.

    복수 정답 지원: "6;182" → [6, 182]  (요약 페이지 + 상세 페이지 모두 정답 인정)
    단일 정수(220, "220")도 그대로 동작한다.
    """
    if isinstance(page_field, (int, float)):
        return [int(page_field)]
    return [int(float(p)) for p in str(page_field).split(";") if str(p).strip()]


def _as_golds(gold):
    """int 하나 또는 페이지 목록을 모두 허용한다 (하위 호환)."""
    return [gold] if isinstance(gold, int) else list(gold)


def page_hit(retrieved_pages, gold):
    """정답 페이지 중 하나라도 검색된 페이지 목록에 있으면 True."""
    golds = _as_golds(gold)
    return any(g in retrieved_pages for g in golds)


def reciprocal_rank(retrieved_pages, gold):
    """정답 페이지가 처음 등장한 순위의 역수(1/rank). 없으면 0.0.

    복수 정답이면 가장 먼저 등장한 정답 기준(최고 순위).
    """
    golds = set(_as_golds(gold))
    for i, p in enumerate(retrieved_pages, start=1):
        if p in golds:
            return 1.0 / i
    return 0.0


def retrieval_error_code(hit):
    """검색 성공은 빈 값, 정답 페이지 검색 실패는 E3으로 분류한다."""
    return "" if hit else "E3"


def get_embedding_model_name(provider, config):
    """임베딩 provider에 대응하는 정확한 모델명을 반환한다."""
    model_names = {
        "huggingface": config.HF_EMBEDDING_MODEL,
        "gemini": config.GEMINI_EMBEDDING_MODEL,
        "openai": config.OPENAI_EMBEDDING_MODEL,
    }
    return model_names[provider]


def get_chunk_count(vectorstore):
    """로드된 Chroma 또는 FAISS 벡터스토어의 실제 청크 수를 반환한다."""
    collection = getattr(vectorstore, "_collection", None)
    if collection is not None and hasattr(collection, "count"):
        return int(collection.count())

    index = getattr(vectorstore, "index", None)
    if index is not None and hasattr(index, "ntotal"):
        return int(index.ntotal)

    raise ValueError("지원하지 않는 벡터스토어: 청크 수를 확인할 수 없습니다.")


import argparse
import re
from collections import Counter
from rag_chain import get_retriever, dedup_docs_by_page, SEARCH_CHOICES, EMBEDDING_CHOICES


def split_normalized_sentences(text):
    """줄바꿈 또는 문장부호 경계로 나누고 공백을 정규화한다."""
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [
        re.sub(r"\s+", " ", part).strip()
        for part in parts
        if part.strip()
    ]


def count_duplicate_sentences(docs):
    """top-k 청크 안에서 동일 문장이 추가로 등장한 횟수를 센다."""
    sentences = []
    for doc in docs:
        sentences.extend(split_normalized_sentences(doc.page_content))

    counts = Counter(sentences)
    return sum(count - 1 for count in counts.values() if count > 1)


def main():
    import pandas as pd

    import config
    from build_vectorstore import load_vectorstore
    from rag_chain import get_retriever, SEARCH_CHOICES, EMBEDDING_CHOICES

    parser = argparse.ArgumentParser(description="검색 품질 채점 (Hit@k, MRR)")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--search-type", choices=SEARCH_CHOICES, default=config.SEARCH_TYPE)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--vectorstore", choices=["chroma", "faiss"], default=config.VECTORSTORE)
    parser.add_argument("--embedding", choices=EMBEDDING_CHOICES, default=config.EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=None,
                        help="모델 ID 직접 지정 (예: intfloat/multilingual-e5-small)")    
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)

    parser.add_argument("--dedup", action="store_true", help="페이지 중복 제거 후 top-k 선별")
    parser.add_argument("--fetch-k", type=int, default=15, help="dedup 시 1차 검색 후보 수")
    parser.add_argument("--mmr-lambda", type=float, default=None, help="MMR: 1=관련성만, 0=다양성만")
    parser.add_argument("--bm25-weight", type=float, default=None, help="hybrid: BM25 비중 (벡터=1-값)")
    args = parser.parse_args()

    questions = pd.read_csv(config.EVAL_DIR / "questions.csv", encoding="utf-8-sig")
    references = pd.read_csv(config.EVAL_DIR / "references.csv", encoding="utf-8-sig")
    df = questions.merge(references, on="id")

    vs = load_vectorstore(args.vectorstore, args.embedding, args.chunk_size, args.overlap,
                          embedding_model=args.embedding_model)
    
    search_k = args.fetch_k if args.dedup else args.top_k
    retriever = get_retriever(
        vs, args.search_type, search_k, args.chunk_size, args.overlap,
        fetch_k=args.fetch_k if args.search_type == "mmr" else None,
        lambda_mult=args.mmr_lambda,
        hybrid_weights=[args.bm25_weight, 1 - args.bm25_weight] if args.bm25_weight is not None else None,
    )
    
    # --embedding-model로 직접 지정한 경우 그 모델 ID를 기록 (§21: 정확한 모델명 기록)
    embedding_model = args.embedding_model or get_embedding_model_name(args.embedding, config)
    chunk_count = get_chunk_count(vs)

    rows = []
    for _, row in df.iterrows():
        docs = retriever.invoke(row["question"])
        if args.dedup:
            docs = dedup_docs_by_page(docs, args.top_k)
        pages = retrieved_physical_pages(docs)

        gold = parse_gold_pages(row["page"])
        hit = page_hit(pages, gold)
        duplicate_sentence_count = count_duplicate_sentences(docs)
        rows.append({
            "id": row["id"], "question": row["question"],
            "embedding_model": embedding_model,
            "chunk_count": chunk_count,
            "search_type": args.search_type,
            "top_k": args.top_k,
            "gold_page": gold, "retrieved_pages": pages,
            "hit": hit,
            "reciprocal_rank": round(reciprocal_rank(pages, gold), 3),
            "error_code": retrieval_error_code(hit),
            "duplicate_sentence_count": duplicate_sentence_count,
        })

    result = pd.DataFrame(rows)
    hit_at_k = result["hit"].mean()
    mrr = result["reciprocal_rank"].mean()
    duplicate_sentence_count = int(result["duplicate_sentence_count"].sum())

    out = config.EVAL_DIR / f"retrieval_{args.run_name}.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n===== 검색 채점: {args.run_name} (top_k={args.top_k}) =====")
    print(f"Hit@{args.top_k}: {hit_at_k:.3f}  |  MRR: {mrr:.3f}")
    print(f"중복 문장 수(top-{args.top_k} 전체): {duplicate_sentence_count}")
    print(f"문항별 결과 -> {out.name}")
    print("\n[실패 문항 - 정답 페이지 못 찾음]")
    fails = result[~result["hit"]]
    if fails.empty:
        print("  (없음)")
    else:
        for _, r in fails.iterrows():
            print(f"  Q{r['id']} (정답 p.{r['gold_page']}): 검색됨 {r['retrieved_pages']} | {r['question'][:30]}")


if __name__ == "__main__":
    main()
