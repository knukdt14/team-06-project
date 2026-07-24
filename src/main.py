"""
연말정산 Q&A 챗봇 — 대화형 CLI.

실행:
    python src/main.py
    python src/main.py --llm upstage --top-k 5
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
    parser.add_argument("--show-context", action="store_true", help="검색된 문맥도 출력")
    args = parser.parse_args()

    print("벡터스토어 로드 중...")
    chain, retriever = build_chain(llm_provider=args.llm, prompt_name=args.prompt,
                                   top_k=args.top_k)

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
