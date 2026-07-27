"""
연말정산 Q&A 챗봇 — 대화형 CLI.

기본값은 최종 확정 파이프라인을 따른다 (bge-m3 + similarity + 페이지 dedup + top_k=3
+ Upstage solar-pro — final_model_selection.md). config.py에서 일괄 변경 가능.

실행:
    python src/main.py
    python src/main.py --llm gemini --top-k 5
    python src/main.py --no-dedup   # dedup 끄고 비교해보고 싶을 때
"""
import argparse

import config
from rag_chain import PROMPTS, build_chain


def main():
    parser = argparse.ArgumentParser(description="연말정산 Q&A 챗봇")
    from rag_chain import LLM_CHOICES
    parser.add_argument("--llm", choices=LLM_CHOICES, default=config.LLM_PROVIDER)
    parser.add_argument("--prompt", choices=list(PROMPTS), default="basic")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=config.DEDUP,
                        help="페이지 중복 제거 (기본 on — 끄려면 --no-dedup)")
    parser.add_argument("--show-context", action="store_true", help="검색된 문맥도 출력")
    args = parser.parse_args()

    print("벡터스토어 로드 중...")
    chain, retriever = build_chain(llm_provider=args.llm, prompt_name=args.prompt,
                                   top_k=args.top_k, dedup=args.dedup)

    print("=" * 60)
    print("국세청 연말정산 Q&A 챗봇 (종료: q)")
    print("예시: 월세 살면 얼마나 돌려받아요? / 자녀 2명이면 세액공제 얼마예요?")
    print("=" * 60)

    while True:
        question = input("\n질문> ").strip()
        if question.lower() in ("q", "quit", "exit"):
            break
        if not question:
            continue

        if args.show_context:
            print("\n[검색된 문맥]")
            for doc in retriever.invoke(question):
                print(f"  - p.{doc.metadata.get('page')}: {doc.page_content[:80]}...")

        print("\n[답변]")
        print(chain.invoke(question))


if __name__ == "__main__":
    main()
