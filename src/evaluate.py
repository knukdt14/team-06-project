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
import time
from datetime import datetime

import pandas as pd

import config
from eval_retrieval import get_embedding_model_name
from rag_chain import build_chain, format_docs

RESULTS_PATH = config.EVAL_DIR / "results.csv"


def load_evalset():
    questions = pd.read_csv(config.EVAL_DIR / "questions.csv", encoding="utf-8-sig")
    references = pd.read_csv(config.EVAL_DIR / "references.csv", encoding="utf-8-sig")
    df = questions.merge(references, on="id")
    print(f"[evaluate] 평가셋 {len(df)}개 로드")
    return df


def run_chain(df, chain, retriever):
    """평가셋 전체에 대해 답변·문맥·문맥 페이지를 수집하고 응답 시간을 측정한다."""

    answers, contexts, latencies = [], [], []
    context_pages_by_id = {}  # {질문 id: [raw 페이지(0-indexed)]} — 인용·검색 채점용
    for _, row in df.iterrows():
        start = time.time()
        docs = retriever.invoke(row["question"])
        answer = chain.invoke(row["question"])
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
    from rag_chain import EMBEDDING_CHOICES, LLM_CHOICES, SEARCH_CHOICES
    parser.add_argument("--llm", choices=LLM_CHOICES, default=config.LLM_PROVIDER)
    parser.add_argument("--prompt", default="basic")
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
    parser.add_argument("--with-ragas", action="store_true")
    args = parser.parse_args()

    df = load_evalset()
    chain, retriever = build_chain(
        vectorstore=args.vectorstore, embedding=args.embedding,
        llm_provider=args.llm, prompt_name=args.prompt,
        search_type=args.search_type, top_k=args.top_k,
        chunk_size=args.chunk_size, overlap=args.overlap,
        embedding_model=args.embedding_model,
        dedup=args.dedup, fetch_k=args.fetch_k,
    )

    answers, contexts, latencies, context_pages_by_id = run_chain(df, chain, retriever)

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
        "search_type": args.search_type, "top_k": args.top_k,
        "dedup": args.dedup, "fetch_k": args.fetch_k if args.dedup else None,
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
