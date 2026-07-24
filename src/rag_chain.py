"""
[담당: 이희영(검색) + 이수홍(LLM/프롬프트)] RAG 체인 구성.

벡터스토어 → retriever → 프롬프트 → LLM 으로 이어지는 LCEL 체인.

실험 파라미터:
- LLM: gemini / openai / upstage / claude / huggingface(공개 모델)
- 프롬프트: basic / simple / reasoning
- 검색: similarity / mmr / hybrid(BM25+벡터), top_k
- 벡터스토어·임베딩·chunk_size·overlap: build_vectorstore 설정 재사용

실행 예:
    python src/rag_chain.py --question "월세 살면 얼마나 돌려받아요?"
    python src/rag_chain.py --question "..." --llm openai --prompt reasoning
    python src/rag_chain.py --question "..." --search-type hybrid --top-k 5
    python src/rag_chain.py --question "..." --llm huggingface
"""
import argparse

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config
from build_vectorstore import load_vectorstore
from load_pdf import get_chunks

load_dotenv()

# ── 프롬프트 ──────────────────────────
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

PROMPT_REASONING = ChatPromptTemplate.from_template("""\
당신은 국세청 연말정산 안내 문서를 기반으로 답변하는 세무 상담 챗봇입니다.

다음 순서로 생각한 뒤 답하세요.
1. [질문]이 묻는 공제 항목이 무엇인지 파악한다.
2. [문맥]에서 해당 항목의 요건·공제율·한도를 찾는다.
3. 찾은 내용만으로 답을 구성한다. 문맥에 없으면 "문서에서 찾을 수 없습니다"라고 답한다.

최종 답변만 출력하고, 근거 페이지를 "(근거: p.OO)" 형식으로 표시하세요.

[문맥]
{context}

[질문]
{question}

[답변]
""")

PROMPTS = {"basic": PROMPT_BASIC, "simple": PROMPT_SIMPLE, "reasoning": PROMPT_REASONING}


def get_llm(provider=config.LLM_PROVIDER):
    """LLM을 생성한다. (실험 파라미터: LLM 모델)"""

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.GEMINI_LLM_MODEL, temperature=0)
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.OPENAI_LLM_MODEL, temperature=0)
    
    if provider == "upstage":
        from langchain_upstage import ChatUpstage
        return ChatUpstage(model=config.UPSTAGE_LLM_MODEL)
    
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.CLAUDE_LLM_MODEL, temperature=0)
    
    if provider == "huggingface":
        from langchain_huggingface import HuggingFacePipeline
        return HuggingFacePipeline.from_model_id(
            model_id=config.HF_LLM_MODEL,
            task=config.HF_LLM_TASK,
            pipeline_kwargs={"max_new_tokens": 256},
        )
    raise ValueError(f"지원하지 않는 LLM: {provider}")


def get_retriever(vs, search_type=config.SEARCH_TYPE, top_k=config.TOP_K,
                  chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """retriever를 생성한다. (실험 파라미터: 유사도 검색 알고리즘, top_k)"""
    if search_type in ("similarity", "mmr"):
        return vs.as_retriever(search_type=search_type, search_kwargs={"k": top_k})

    if search_type == "hybrid":
        # BM25(키워드) + 벡터 앙상블 검색
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        chunks = get_chunks(chunk_size, overlap)
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = top_k
        vector = vs.as_retriever(search_kwargs={"k": top_k})
        return EnsembleRetriever(retrievers=[bm25, vector], weights=config.HYBRID_WEIGHTS)

    raise ValueError(f"지원하지 않는 검색 방식: {search_type}")


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
    retriever = get_retriever(vs, search_type, top_k, chunk_size, overlap)
    llm = get_llm(llm_provider)
    prompt = PROMPTS[prompt_name]

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


LLM_CHOICES = ["gemini", "openai", "upstage", "claude", "huggingface"]
SEARCH_CHOICES = ["similarity", "mmr", "hybrid"]
EMBEDDING_CHOICES = ["huggingface", "openai", "gemini"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 질의응답")
    parser.add_argument("--question", required=True)
    parser.add_argument("--llm", choices=LLM_CHOICES, default=config.LLM_PROVIDER)
    parser.add_argument("--prompt", choices=list(PROMPTS), default="basic")
    parser.add_argument("--search-type", choices=SEARCH_CHOICES, default=config.SEARCH_TYPE)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--vectorstore", choices=["chroma", "faiss"], default=config.VECTORSTORE)
    parser.add_argument("--embedding", choices=EMBEDDING_CHOICES, default=config.EMBEDDING_PROVIDER)
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
