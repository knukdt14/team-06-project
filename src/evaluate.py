"""
[담당: 이수홍] RAG 성능 평가.

eval/questions.csv + references.csv의 평가셋으로 RAG 체인을 실행하고
- BERTScore (필수)
- 규칙 기반 채점 (scoring.py): 핵심 숫자 정확도·조건 포함률·페이지 인용 정확도·
  거절 정확도 + 오류 코드(E1~E11 중 자동 판정분) 태깅
- 응답 시간
- RAGAS (선택: --with-ragas, OpenAI 키 필요)
를 계산하여 eval/results.csv에 실험 설정과 함께 누적 기록한다.

BERTScore만으로는 "13만원을 7만원이라 답한 오답"을 못 잡으므로,
최적 설정 선택은 numeric_acc(핵심 숫자 정확도)를 우선 기준으로 본다.

실행 예:
    python src/evaluate.py --run-name baseline
    python src/evaluate.py --run-name cs500 --chunk-size 500 --overlap 100
    python src/evaluate.py --run-name upstage_k5 --llm upstage --top-k 5
    python src/evaluate.py --run-name baseline_ragas --with-ragas

※ 이미 실행한 detail_*.csv를 새 지표로 다시 채점만 하려면 (API 불필요):
    python src/rescore.py --detail eval/detail_baseline.csv --retrieval eval/retrieval_baseline.csv
"""
import argparse
import sys
import time
from datetime import datetime

import pandas as pd

# ※ 버그 수정: LLM(Upstage)이 법령 인용에 반각 괄호(｢｣) 등을 자주 쓰는데,
# Windows 콘솔 기본 인코딩(cp949)이 이를 못 담아 print()에서 UnicodeEncodeError로
# 평가 도중 죽는 문제가 있었다 (백그라운드 실행 시 재현). stdout을 UTF-8로
# 강제해 인코딩 실패로 전체 평가가 중단되지 않게 한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
from eval_retrieval import get_embedding_model_name
from rag_chain import PROMPTS, format_docs, get_llm

RESULTS_PATH = config.EVAL_DIR / "results.csv"


def load_evalset():
    questions = pd.read_csv(config.EVAL_DIR / "questions.csv", encoding="utf-8-sig")
    references = pd.read_csv(config.EVAL_DIR / "references.csv", encoding="utf-8-sig")
    df = questions.merge(references, on="id")
    print(f"[evaluate] 평가셋 {len(df)}개 로드")
    return df


def load_query_rewrites(df, rewrite_llm):
    """캐시된 질의 재작성(eval/query_rewrites.csv)을 읽는다. 없는 id는 새로 생성해 캐시에 더한다.

    eval_retrieval.py --rewrite와 동일한 캐시 파일을 공유한다 (중복 LLM 호출 방지).
    """
    from rag_chain import rewrite_query

    cache_path = config.EVAL_DIR / "query_rewrites.csv"
    rewrites = {}
    if cache_path.exists():
        cached = pd.read_csv(cache_path, encoding="utf-8-sig")
        rewrites = dict(zip(cached["id"], cached["rewritten"]))

    missing = [(r["id"], r["question"]) for _, r in df.iterrows() if r["id"] not in rewrites]
    if missing:
        print(f"[evaluate] 확장 질문 {len(missing)}개 생성 중 (LLM: {rewrite_llm})...")
        for qid, q in missing:
            rewrites[qid] = rewrite_query(q, rewrite_llm)
        pd.DataFrame(
            [{"id": qid, "question": q, "rewritten": rewrites[qid], "llm": rewrite_llm}
             for qid, q in zip(df["id"], df["question"])]
        ).to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"[evaluate] 캐시 저장 -> {cache_path.name}")
    return rewrites


def retrieve_docs(question, qid, retriever, top_k, rewrite, rewrites, rerank, rerank_model):
    """한 질문에 대해 (rewrite+RRF →) (rerank →) dedup top_k 문서를 반환한다.

    eval_retrieval.py의 검색 파이프라인(rewrite+RRF 앙상블, cross-encoder rerank)을
    답변 생성(evaluate.py)에도 그대로 적용해, 희영님이 팀장님께 요청한 "cs300
    dedup만 / rewrite+rerank" 2안의 실제 답변 품질(numeric_acc 등)과 총 응답시간을
    잰다. 검색 로직 자체는 이식이 아니라 재사용(rag_chain의 rrf_merge/rerank_docs)이라
    두 스크립트의 검색 결과가 어긋날 위험이 없다.
    """
    from rag_chain import dedup_docs_by_page, rerank_docs, rrf_merge

    if rewrite:
        docs = rrf_merge([retriever.invoke(question), retriever.invoke(rewrites[qid])])
    else:
        docs = retriever.invoke(question)

    if rerank:
        docs = rerank_docs(question, docs, model_name=rerank_model)

    if rewrite or rerank:
        docs = dedup_docs_by_page(docs, top_k)
    return docs


def run_chain(df, answer_chain, retriever, top_k, rewrite=False, rewrites=None,
             rerank=False, rerank_model=config.RERANKER_MODEL):
    """평가셋 전체에 대해 답변·문맥·문맥 페이지를 수집하고 응답 시간을 측정한다.

    latency는 검색(rewrite+RRF/rerank 포함) + LLM 호출을 합친 총 응답시간이다.
    """

    answers, contexts, latencies = [], [], []
    context_pages_by_id = {}  # {질문 id: [raw 페이지(0-indexed)]} — 인용·검색 채점용
    for _, row in df.iterrows():
        start = time.time()
        docs = retrieve_docs(row["question"], row["id"], retriever, top_k,
                             rewrite, rewrites, rerank, rerank_model)
        answer = answer_chain.invoke({"context": format_docs(docs), "question": row["question"]})
        latencies.append(time.time() - start)
        answers.append(answer)
        contexts.append([d.page_content for d in docs])
        context_pages_by_id[row["id"]] = [d.metadata.get("page", -1) for d in docs]
        print(f"  Q: {row['question'][:40]}... ({latencies[-1]:.1f}s)")
        print(f"  A: {answer}")
    return answers, contexts, latencies, context_pages_by_id


def compute_bertscore(answers, references):
    """BERTScore P/R/F1 (필수 지표)."""

    from bert_score import score
    P, R, F1 = score(
        answers, references, lang="ko",
        model_type="bert-base-multilingual-cased",
    )
    return P.tolist(), R.tolist(), F1.tolist()


def compute_ragas(questions, answers, contexts, references):
    """RAGAS 지표 (선택). evaluator LLM으로 OpenAI 키가 필요하다."""
    
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": references,
    })
    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result.to_pandas()


def main():
    parser = argparse.ArgumentParser(description="RAG 평가")
    parser.add_argument("--run-name", required=True, help="실험 이름 (results.csv에 기록)")
    from rag_chain import EMBEDDING_CHOICES, LLM_CHOICES, SEARCH_CHOICES, get_retriever
    from build_vectorstore import load_vectorstore
    from langchain_core.output_parsers import StrOutputParser

    parser.add_argument("--llm", choices=LLM_CHOICES, default=config.LLM_PROVIDER)
    parser.add_argument("--prompt", choices=list(PROMPTS), default=config.PROMPT_NAME)
    parser.add_argument("--search-type", choices=SEARCH_CHOICES, default=config.SEARCH_TYPE)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--vectorstore", choices=["chroma", "faiss"], default=config.VECTORSTORE)
    parser.add_argument("--embedding", choices=EMBEDDING_CHOICES, default=config.EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=None,
                        help="모델 ID 직접 지정 (예: intfloat/multilingual-e5-small) — 이희영 임베딩 비교용")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    # 기본값이 최종 확정 설정(config.DEDUP=True)을 따른다 — 끄려면 --no-dedup
    parser.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=config.DEDUP,
                        help="페이지 중복 제거 후 top-k 선별 (이희영, dedup_effect.md: bge-m3 Hit@3 0.923→1.000)")
    parser.add_argument("--fetch-k", type=int, default=config.DEDUP_FETCH_K, help="dedup 시 1차 검색 후보 수")
    parser.add_argument("--rewrite", action="store_true",
                        help="query rewrite + RRF 앙상블 (final66_search_comparison.md 최종 채택 후보)")
    parser.add_argument("--rewrite-llm", choices=LLM_CHOICES, default=config.LLM_PROVIDER,
                        help="질문 재작성에 쓸 LLM. eval/query_rewrites.csv 캐시 재사용")
    parser.add_argument("--rerank", action="store_true",
                        help="cross-encoder 재정렬 (final66_search_comparison.md 최종 채택 후보)")
    parser.add_argument("--rerank-model", default=config.RERANKER_MODEL)
    parser.add_argument("--with-ragas", action="store_true")
    args = parser.parse_args()

    df = load_evalset()

    vs = load_vectorstore(args.vectorstore, args.embedding, args.chunk_size, args.overlap,
                         embedding_model=args.embedding_model)
    wide_mode = args.rewrite or args.rerank
    retriever = get_retriever(
        vs, args.search_type,
        args.fetch_k if wide_mode else args.top_k,
        args.chunk_size, args.overlap,
        dedup=args.dedup and not wide_mode, fetch_k=args.fetch_k,
    )
    rewrites = load_query_rewrites(df, args.rewrite_llm) if args.rewrite else None

    answer_chain = PROMPTS[args.prompt] | get_llm(args.llm) | StrOutputParser()
    answers, contexts, latencies, context_pages_by_id = run_chain(
        df, answer_chain, retriever, args.top_k,
        rewrite=args.rewrite, rewrites=rewrites,
        rerank=args.rerank, rerank_model=args.rerank_model,
    )

    print("[evaluate] BERTScore 계산 중...")
    P, R, F1 = compute_bertscore(answers, df["reference"].tolist())

    # 문항별 상세 결과 저장
    detail = df.copy()
    detail["answer"] = answers
    detail["latency_sec"] = [round(x, 2) for x in latencies]
    detail["bertscore_p"] = [round(x, 4) for x in P]
    detail["bertscore_r"] = [round(x, 4) for x in R]
    detail["bertscore_f1"] = [round(x, 4) for x in F1]

    # 규칙 기반 채점 (핵심 숫자·조건·인용·거절 + 오류 코드)
    print("[evaluate] 규칙 기반 채점 중 (scoring.py)...")
    from scoring import score_dataframe
    detail, rule_summary = score_dataframe(detail, context_pages_by_id)

    detail_path = config.EVAL_DIR / f"detail_{args.run_name}.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"[evaluate] 문항별 결과 → {detail_path.name}")

    # 실험 요약을 results.csv에 누적
    summary = {
        "run_name": args.run_name,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "llm": args.llm, "prompt": args.prompt,
        # §21: provider명이 아니라 정확한 모델 ID로 기록
        "embedding": args.embedding_model or get_embedding_model_name(args.embedding, config),
        "vectorstore": args.vectorstore,
        "chunk_size": args.chunk_size, "overlap": args.overlap,
        "search_type": (args.search_type
                       + ("+rewrite_rrf" if args.rewrite else "")
                       + ("+rerank" if args.rerank else "")),
        "top_k": args.top_k,
        "dedup": args.dedup, "fetch_k": args.fetch_k if (args.dedup or args.rewrite or args.rerank) else None,
        "bertscore_f1": round(sum(F1) / len(F1), 4),
        "avg_latency_sec": round(sum(latencies) / len(latencies), 2),
        **rule_summary,  # numeric_acc, condition_recall, citation_acc, hit_rate,
                         # wrong_refusals, abstention_acc, error_counts 등
    }

    if args.with_ragas:
        print("[evaluate] RAGAS 계산 중...")
        df_ragas = compute_ragas(df["question"].tolist(), answers, contexts,
                                 df["reference"].tolist())
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric in df_ragas.columns:
                summary[metric] = round(df_ragas[metric].mean(), 4)

    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig")
        results = pd.concat([results, pd.DataFrame([summary])], ignore_index=True)
    else:
        results = pd.DataFrame([summary])
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    print("\n===== 실험 결과 =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    flagged = detail[detail["error_codes"].astype(str).str.len() > 0]
    if not flagged.empty:
        print("\n[오류 태깅된 문항]")
        for _, r in flagged.iterrows():
            miss = r["missing_numbers"] or r["missing_conditions"]
            print(f"  Q{r['id']} [{r['error_codes']}] 누락: {miss} | {r['question'][:30]}")

    print(f"→ {RESULTS_PATH.name}에 누적 저장 완료")


if __name__ == "__main__":
    main()
