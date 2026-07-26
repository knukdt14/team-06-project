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


import argparse


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
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    questions = pd.read_csv(config.EVAL_DIR / "questions.csv", encoding="utf-8-sig")
    references = pd.read_csv(config.EVAL_DIR / "references.csv", encoding="utf-8-sig")
    df = questions.merge(references, on="id")

    vs = load_vectorstore(args.vectorstore, args.embedding, args.chunk_size, args.overlap)
    retriever = get_retriever(vs, args.search_type, args.top_k, args.chunk_size, args.overlap)

    rows = []
    for _, row in df.iterrows():
        docs = retriever.invoke(row["question"])
        pages = retrieved_physical_pages(docs)
        gold = int(row["page"])
        rows.append({
            "id": row["id"], "question": row["question"],
            "gold_page": gold, "retrieved_pages": pages,
            "hit": page_hit(pages, gold),
            "reciprocal_rank": round(reciprocal_rank(pages, gold), 3),
        })

    result = pd.DataFrame(rows)
    hit_at_k = result["hit"].mean()
    mrr = result["reciprocal_rank"].mean()

    out = config.EVAL_DIR / f"retrieval_{args.run_name}.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n===== 검색 채점: {args.run_name} (top_k={args.top_k}) =====")
    print(f"Hit@{args.top_k}: {hit_at_k:.3f}  |  MRR: {mrr:.3f}")
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
