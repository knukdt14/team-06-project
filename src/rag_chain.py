"""
[담당: 이희영(검색) + 이수홍(LLM/프롬프트)] RAG 체인 구성.

벡터스토어 → retriever → 프롬프트 → LLM 으로 이어지는 LCEL 체인.
- 검색 방식(similarity/mmr), top_k 조절 가능      (실험 파라미터 3: 검색)
- LLM(OpenAI/Upstage), 프롬프트 교체 가능          (실험 파라미터 4: LLM/프롬프트)
- 답변에 근거 페이지를 함께 반환

실행 예:
    python src/rag_chain.py --question "월세 살면 얼마나 돌려받아요?"
    python src/rag_chain.py --question "자녀가 2명이면 세액공제 얼마?" --llm upstage --top-k 5
"""
import argparse

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config
from build_vectorstore import load_vectorstore

load_dotenv()

# ── 프롬프트 (실험: 프롬프트를 바꿔가며 성능 비교) ──────────────
PROMPT_BASIC = ChatPromptTemplate.from_template("""\
당신은 국세청 연말정산 안내 문서를 기반으로 답변하는 세무 상담 챗봇입니다.

다음 [문맥]만을 근거로 [질문]에 답하세요.
- 문맥에 없는 내용은 "문서에서 찾을 수 없습니다"라고 답하세요.
- 금액, 공제율, 한도는 문맥에 있는 숫자를 정확히 인용하세요.
- 답변 마지막에 근거가 된 문서 페이지를 "(근거: p.OO)" 형식으로 표시하세요.

[문맥]
{context}

[질문]
{question}

[답변]
""")

PROMPT_SIMPLE = ChatPromptTemplate.from_template("""\
[문맥]을 참고하여 [질문]에 답하세요.

[문맥]
{context}

[질문]
{question}
""")

PROMPTS = {"basic": PROMPT_BASIC, "simple": PROMPT_SIMPLE}


def get_llm(provider=config.LLM_PROVIDER):
    """LLM을 생성한다."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.GEMINI_LLM_MODEL, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.OPENAI_LLM_MODEL, temperature=0)
    if provider == "upstage":
        from langchain_upstage import ChatUpstage
        return ChatUpstage(model=config.UPSTAGE_LLM_MODEL)
    raise ValueError(f"지원하지 않는 LLM: {provider}")


def format_docs(docs):
    """검색된 청크를 페이지 정보와 함께 하나의 문자열로 합친다."""
    return "\n\n".join(
        f"(p.{doc.metadata.get('page', '?')}) {doc.page_content}" for doc in docs
    )


def build_chain(vectorstore=config.VECTORSTORE,
                embedding=config.EMBEDDING_PROVIDER,
                llm_provider=config.LLM_PROVIDER,
                prompt_name="basic",
                search_type=config.SEARCH_TYPE,
                top_k=config.TOP_K,
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP):
    """RAG 체인과 retriever를 생성해 (chain, retriever)로 반환한다."""
    vs = load_vectorstore(vectorstore, embedding, chunk_size, overlap)
    retriever = vs.as_retriever(search_type=search_type, search_kwargs={"k": top_k})
    llm = get_llm(llm_provider)
    prompt = PROMPTS[prompt_name]

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 질의응답")
    parser.add_argument("--question", required=True)
    parser.add_argument("--llm", choices=["gemini", "upstage", "openai"], default=config.LLM_PROVIDER)
    parser.add_argument("--prompt", choices=list(PROMPTS), default="basic")
    parser.add_argument("--search-type", choices=["similarity", "mmr"], default=config.SEARCH_TYPE)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--vectorstore", choices=["chroma", "faiss"], default=config.VECTORSTORE)
    parser.add_argument("--embedding", choices=["gemini", "huggingface", "openai"], default=config.EMBEDDING_PROVIDER)
    args = parser.parse_args()

    chain, retriever = build_chain(
        vectorstore=args.vectorstore, embedding=args.embedding,
        llm_provider=args.llm, prompt_name=args.prompt,
        search_type=args.search_type, top_k=args.top_k,
    )

    print("[검색된 문맥]")
    for doc in retriever.invoke(args.question):
        print(f"  - p.{doc.metadata.get('page')}: {doc.page_content[:80]}...")

    print("\n[답변]")
    print(chain.invoke(args.question))
