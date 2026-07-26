"""Chroma/FAISS 엔진과 FAISS Flat/HNSW 인덱스를 공정하게 비교한다.

문서 청크와 임베딩을 한 번만 만들고 세 저장 전략에 동일한 벡터를 넣는다.
따라서 build_time은 PDF 로드·청킹·임베딩 시간을 제외한 인덱스 생성 및 저장
시간이며, search_time은 질의 임베딩 시간을 제외한 벡터 검색 시간이다.
"""

import argparse
import gc
import statistics
import time
from pathlib import Path

import pandas as pd

import config
from build_vectorstore import create_faiss_from_embeddings, get_embeddings
from eval_retrieval import (
    count_duplicate_sentences,
    page_hit,
    parse_gold_pages,
    reciprocal_rank,
    retrieval_error_code,
    retrieved_physical_pages,
)
from load_pdf import get_chunks


VARIANTS = ("chroma", "faiss_flat", "faiss_hnsw")
LOAD_REPEATS = 5
SEARCH_REPEATS = 30


def directory_size_bytes(path):
    """디렉터리 아래 일반 파일의 바이트 합계를 반환한다."""

    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def build_store(name, path, texts, vectors, metadatas, ids, embeddings):
    """사전 계산한 동일 벡터로 한 저장 전략을 만들고 저장한다."""

    if name == "chroma":
        from langchain_chroma import Chroma

        vectorstore = Chroma(
            collection_name="langchain",
            embedding_function=embeddings,
            persist_directory=str(path),
        )
        vectorstore._collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        return vectorstore

    faiss_index = name.removeprefix("faiss_")
    vectorstore = create_faiss_from_embeddings(
        zip(texts, vectors),
        embeddings,
        metadatas,
        ids=ids,
        faiss_index=faiss_index,
    )
    vectorstore.save_local(str(path))
    return vectorstore


def load_store(name, path, embeddings):
    """디스크에서 저장 전략별 벡터스토어를 로드한다."""

    if name == "chroma":
        from langchain_chroma import Chroma

        return Chroma(
            collection_name="langchain",
            embedding_function=embeddings,
            persist_directory=str(path),
        )

    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def measure_load(name, path, embeddings):
    """동일 프로세스에서 객체 재구성 시간을 반복 측정해 중앙값을 반환한다."""

    samples = []
    vectorstore = None
    for _ in range(LOAD_REPEATS):
        gc.collect()
        started = time.perf_counter()
        vectorstore = load_store(name, path, embeddings)
        samples.append(time.perf_counter() - started)
    return vectorstore, statistics.median(samples), samples


def evaluate_quality(name, vectorstore, df, query_vectors, top_k, chunk_count):
    """사전 계산한 질의 벡터로 Hit@k/MRR과 문항별 결과를 계산한다."""

    rows = []
    for (_, row), query_vector in zip(df.iterrows(), query_vectors):
        docs = vectorstore.similarity_search_by_vector(query_vector, k=top_k)
        pages = retrieved_physical_pages(docs)
        gold = parse_gold_pages(row["page"])
        hit = page_hit(pages, gold)
        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "variant": name,
                "chunk_count": chunk_count,
                "top_k": top_k,
                "gold_page": gold,
                "retrieved_pages": pages,
                "hit": hit,
                "reciprocal_rank": round(reciprocal_rank(pages, gold), 3),
                "error_code": retrieval_error_code(hit),
                "duplicate_sentence_count": count_duplicate_sentences(docs),
            }
        )
    result = pd.DataFrame(rows)
    return result, float(result["hit"].mean()), float(result["reciprocal_rank"].mean())


def measure_search(vectorstore, query_vectors, top_k):
    """질의 임베딩을 제외한 similarity_search_by_vector 평균 시간을 잰다."""

    for query_vector in query_vectors:
        vectorstore.similarity_search_by_vector(query_vector, k=top_k)

    samples = []
    for _ in range(SEARCH_REPEATS):
        for query_vector in query_vectors:
            started = time.perf_counter()
            vectorstore.similarity_search_by_vector(query_vector, k=top_k)
            samples.append(time.perf_counter() - started)
    return statistics.mean(samples), statistics.median(samples), samples


def main():
    parser = argparse.ArgumentParser(
        description="Chroma vs FAISS Flat/HNSW 저장 전략 벤치마크"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    artifacts_root = config.ARTIFACTS_DIR.resolve()
    if output_root == artifacts_root or artifacts_root not in output_root.parents:
        raise ValueError(f"output-root는 {artifacts_root} 아래여야 합니다.")
    if output_root.exists():
        raise FileExistsError(
            f"기존 측정물을 보호하기 위해 이미 존재하는 경로는 사용하지 않습니다: {output_root}"
        )
    output_root.mkdir(parents=True)

    print("[benchmark] header-only PDF 로드·청킹")
    prepare_started = time.perf_counter()
    chunks = get_chunks(args.chunk_size, args.overlap)
    prepare_seconds = time.perf_counter() - prepare_started
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk-{i}" for i in range(len(chunks))]

    embeddings = get_embeddings("huggingface")
    embed_started = time.perf_counter()
    vectors = embeddings.embed_documents(texts)
    embed_seconds = time.perf_counter() - embed_started

    questions = pd.read_csv(config.EVAL_DIR / "questions.csv", encoding="utf-8-sig")
    references = pd.read_csv(config.EVAL_DIR / "references.csv", encoding="utf-8-sig")
    df = questions.merge(references, on="id")
    query_vectors = embeddings.embed_documents(df["question"].tolist())

    summaries = []
    for name in VARIANTS:
        path = output_root / name
        print(f"[benchmark] build {name}")
        build_started = time.perf_counter()
        vectorstore = build_store(
            name, path, texts, vectors, metadatas, ids, embeddings
        )
        build_seconds = time.perf_counter() - build_started
        del vectorstore
        gc.collect()

        vectorstore, load_seconds, load_samples = measure_load(
            name, path, embeddings
        )
        result, hit_at_k, mrr = evaluate_quality(
            name, vectorstore, df, query_vectors, args.top_k, len(chunks)
        )
        result.to_csv(
            config.EVAL_DIR / f"retrieval_store_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        search_mean, search_median, search_samples = measure_search(
            vectorstore, query_vectors, args.top_k
        )
        summaries.append(
            {
                "variant": name,
                "engine": "Chroma" if name == "chroma" else "FAISS",
                "index": "HNSW (Chroma default)"
                if name == "chroma"
                else name.removeprefix("faiss_").upper(),
                "chunk_count": len(chunks),
                "hit_at_3": round(hit_at_k, 3),
                "mrr": round(mrr, 3),
                "build_time_sec": round(build_seconds, 6),
                "load_time_median_ms": round(load_seconds * 1000, 3),
                "search_time_mean_ms": round(search_mean * 1000, 3),
                "search_time_median_ms": round(search_median * 1000, 3),
                "disk_bytes": directory_size_bytes(path),
                "disk_mib": round(directory_size_bytes(path) / (1024 ** 2), 3),
                "load_repeats": len(load_samples),
                "search_queries": len(search_samples),
            }
        )
        del vectorstore

    summary = pd.DataFrame(summaries)
    summary["pdf_chunk_prepare_sec"] = round(prepare_seconds, 6)
    summary["document_embedding_sec"] = round(embed_seconds, 6)
    summary.to_csv(
        config.EVAL_DIR / "store_benchmark.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 저장 전략 비교 =====")
    print(summary.to_string(index=False))
    print(f"\n결과 CSV: {config.EVAL_DIR / 'store_benchmark.csv'}")
    print(f"스토어: {output_root}")


if __name__ == "__main__":
    main()
