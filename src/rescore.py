"""[담당: 이수홍] 기존 detail_*.csv 재채점 (LLM/API 불필요 → Gemini 쿼터 절약).

이미 실행된 평가 결과(answer 포함)를 새 규칙 기반 지표로 다시 채점한다:
핵심 숫자 정확도 / 조건 포함률 / 페이지 인용 정확도 / 거절 여부 / 오류 코드.

검색 페이지 정보(retrieval_*.csv, 이수민 제작)가 있으면 --retrieval로 병합해
검색 실패(E3)와 인용 정확도(E9)까지 채점한다.

실행 예:
    python src/rescore.py --detail eval/detail_baseline.csv --retrieval eval/retrieval_baseline.csv
    python src/rescore.py --detail eval/detail_cs500.csv --retrieval eval/retrieval_cs500.csv

출력: eval/detail_<이름>_scored.csv + 콘솔 요약.
"""
import argparse
import ast
from pathlib import Path

import pandas as pd

import config
from scoring import score_dataframe


def load_context_pages(retrieval_csv):
    """retrieval_*.csv의 retrieved_pages(물리, 1-indexed)를 raw(0-indexed)로 변환.

    반환: {질문 id: [raw 페이지들]}
    """
    df = pd.read_csv(retrieval_csv, encoding="utf-8-sig")
    out = {}
    for _, r in df.iterrows():
        physical = ast.literal_eval(str(r["retrieved_pages"]))
        out[r["id"]] = [p - 1 for p in physical if p >= 1]
    return out


def main():
    parser = argparse.ArgumentParser(description="기존 평가 결과 재채점 (API 불필요)")
    parser.add_argument("--detail", required=True, help="detail_*.csv 경로")
    parser.add_argument("--retrieval", help="retrieval_*.csv 경로 (검색·인용 채점용, 선택)")
    args = parser.parse_args()

    detail = pd.read_csv(args.detail, encoding="utf-8-sig")

    # 예전 detail에는 key_numbers 등이 없으므로 references.csv에서 병합
    refs = pd.read_csv(config.EVAL_DIR / "references.csv", encoding="utf-8-sig")
    for col in ("key_numbers", "key_conditions", "answerable"):
        if col not in detail.columns and col in refs.columns:
            detail = detail.merge(refs[["id", col]], on="id", how="left")

    context_pages = load_context_pages(args.retrieval) if args.retrieval else None
    scored, summary = score_dataframe(detail, context_pages)

    src = Path(args.detail)
    out = src.with_name(src.stem + "_scored.csv")
    scored.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n===== 재채점: {src.name} =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    flagged = scored[scored["error_codes"].astype(str).str.len() > 0]
    if flagged.empty:
        print("\n[오류 태깅된 문항] (없음)")
    else:
        print("\n[오류 태깅된 문항]")
        for _, r in flagged.iterrows():
            miss = r["missing_numbers"] or r["missing_conditions"]
            print(f"  Q{r['id']} [{r['error_codes']}] 누락: {miss} | {str(r['question'])[:30]}")
    print(f"\n문항별 결과 -> {out.name}")


if __name__ == "__main__":
    main()
