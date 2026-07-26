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
    args = parser.parse_args()

    questions = pd.read_csv(config.EVAL_DIR / "questions.csv", encoding="utf-8-sig")
    references = pd.read_csv(config.EVAL_DIR / "references.csv", encoding="utf-8-sig")
    df = questions.merge(references, on="id")

    vs = load_vectorstore(args.vectorstore, args.embedding, args.chunk_size, args.overlap,
                          embedding_model=args.embedding_model)
    retriever = get_retriever(vs, args.search_type, args.top_k, args.chunk_size, args.overlap)
    embedding_model = get_embedding_model_name(args.embedding, config)
    chunk_count = get_chunk_count(vs)

    rows = []
    for _, row in df.iterrows():
        docs = retriever.invoke(row["question"])
        pages = retrieved_physical_pages(docs)
        gold = int(row["page"])
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
