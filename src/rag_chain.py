"""
[담당: 이희영(검색) + 이수홍(LLM/프롬프트)] RAG 체인 구성.

벡터스토어 → retriever → 프롬프트 → LLM 으로 이어지는 LCEL 체인.

실험 파라미터:
- LLM: gemini / openai / upstage / claude / qwen / exaone(HuggingFace 공개 모델, 로컬 실행)
- 프롬프트: basic / simple / reasoning
- 검색: similarity / mmr / hybrid(BM25+벡터), top_k
- 벡터스토어·임베딩·chunk_size·overlap: build_vectorstore 설정 재사용

실행 예:
    python src/rag_chain.py --question "월세 살면 얼마나 돌려받아요?"
    python src/rag_chain.py --question "..." --llm openai --prompt reasoning
    python src/rag_chain.py --question "..." --search-type hybrid --top-k 5
    python src/rag_chain.py --question "..." --llm qwen
    python src/rag_chain.py --question "..." --llm exaone
"""
import argparse
import re

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
        # temperature=0 누락 발견: gemini/openai는 있었는데 upstage만 없어서
        # 완전히 동일한 문맥을 줘도 답변이 매번 달라지는 재현성 문제가 있었다
        # (llm_bgem3_upstage_dedup 재채점 중 Q6에서 실측: retrieved_pages 동일,
        # numeric_score 1.0→0.333 — dedup과 무관한 순수 샘플링 변동으로 확인).
        return ChatUpstage(model=config.UPSTAGE_LLM_MODEL, temperature=0)
    
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.CLAUDE_LLM_MODEL, temperature=0)
    
    if provider in ("qwen", "exaone"):
        # HuggingFace 공개 모델, 로컬 실행(API 키 불필요). ChatHuggingFace로 감싸
        # 각 모델의 tokenizer chat template(instruct 형식)을 그대로 적용한다 —
        # 그냥 HuggingFacePipeline만 쓰면 프롬프트가 일반 텍스트로 이어붙여져
        # instruct 모델 성능이 떨어진다.
        #
        # ※ HuggingFacePipeline.from_model_id(pipeline_kwargs=...)로 만들면
        # ChatHuggingFace와 조합했을 때 return_full_text=False가 무시되어
        # 답변에 채팅 템플릿 전문(system/user 턴 포함)이 그대로 섞여 나온다
        # (실측 확인). transformers pipeline을 직접 만들어 넘기면 정상 동작한다.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import pipeline as hf_pipeline
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        model_id = config.HF_QWEN_LLM_MODEL if provider == "qwen" else config.HF_EXAONE_LLM_MODEL
        # EXAONE은 transformers에 아직 네이티브로 등록돼 있지 않아 모델 저장소의
        # 자체 모델링 코드(trust_remote_code)와 4.x 호환 revision 고정이 필요하다.
        # Qwen은 네이티브 지원이라 둘 다 불필요.
        extra_kwargs = {}
        if provider == "exaone":
            extra_kwargs = {"trust_remote_code": True, "revision": config.HF_EXAONE_REVISION}

        tokenizer = AutoTokenizer.from_pretrained(model_id, **extra_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_id, **extra_kwargs)
        pipe = hf_pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=config.HF_LLM_MAX_NEW_TOKENS, do_sample=False,
            return_full_text=False,
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe), tokenizer=tokenizer)
    raise ValueError(f"지원하지 않는 LLM: {provider}")


def get_retriever(vs, search_type=config.SEARCH_TYPE, top_k=config.TOP_K,
                  chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP,
                  dedup=False, fetch_k=None, lambda_mult=None, hybrid_weights=None):
    """retriever를 생성한다.

    (실험 파라미터: 검색 알고리즘, top_k, MMR fetch_k/lambda, hybrid 가중치,
    페이지 dedup — dedup_effect.md, search_method_comparison.md)

    ※ 복구 메모: main↔yeong 병합 과정에서 dedup 지원이 통째로 빠지고 죽은 코드
    (미도달 분기, 정의되지 않은 search_k/base 참조)가 남아 evaluate.py --dedup이
    TypeError로 즉시 실패하는 상태였다. 이희영의 MMR fetch_k/lambda_mult,
    hybrid_weights는 그대로 살리고 dedup 지원만 복구했다.

    dedup=True면 fetch_k개를 1차로 넉넉히 검색한 뒤 서로 다른 페이지 top_k개만
    남긴다(page 단위, similarity/mmr/hybrid 공통 적용). dedup=True일 때는 MMR
    자체의 fetch_k(다양성 후보 풀)는 사용하지 않고 dedup의 fetch_k가 우선한다
    (두 기능을 동시에 쓰는 실험은 아직 없어 단순화).
    """
    search_k = fetch_k if (dedup and fetch_k) else top_k

    if search_type == "similarity":
        base = vs.as_retriever(search_type="similarity", search_kwargs={"k": search_k})
    elif search_type == "mmr":
        kwargs = {"k": search_k}
        if fetch_k is not None and not dedup:
            kwargs["fetch_k"] = fetch_k          # MMR 자체 후보 풀 (dedup 미사용 시)
        if lambda_mult is not None:
            kwargs["lambda_mult"] = lambda_mult  # 1=관련성만, 0=다양성만
        base = vs.as_retriever(search_type="mmr", search_kwargs=kwargs)
    elif search_type == "hybrid":
        # BM25(키워드) + 벡터 앙상블 검색
        from langchain_community.retrievers import BM25Retriever
        from langchain_classic.retrievers import EnsembleRetriever
        chunks = get_chunks(chunk_size, overlap)
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = search_k
        vector = vs.as_retriever(search_kwargs={"k": search_k})
        base = EnsembleRetriever(retrievers=[bm25, vector],
                                 weights=hybrid_weights or config.HYBRID_WEIGHTS)
    else:
        raise ValueError(f"지원하지 않는 검색 방식: {search_type}")

    if not dedup:
        return base

    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(lambda question: dedup_docs_by_page(base.invoke(question), top_k))


def dedup_docs_by_page(docs, k):
    """[담당: 이희영] 같은 페이지 청크는 1개만 남기고, 서로 다른 페이지 k개를 반환한다.
    (handoff_ensemble.md 제안 ①: 같은 페이지가 top-k를 독점하는 문제 해결)"""
    seen_pages = set()
    kept = []
    for d in docs:
        page = d.metadata.get("page")
        if page in seen_pages:
            continue
        seen_pages.add(page)
        kept.append(d)
        if len(kept) == k:
            break
    return kept


REWRITE_PROMPT = ChatPromptTemplate.from_template("""\
다음 질문을 국세청 연말정산 안내 문서에서 검색하기 좋게 한 문장으로 재작성하세요.
- 질문이 가리키는 공제·감면 제도의 공식 명칭을 포함하세요.
- 문서에 쓰이는 세무 용어(세액공제, 소득공제, 한도, 공제율 등)를 보강하세요.
- 출력은 재작성된 질문 문장 하나뿐이어야 합니다.
- 설명, 풀이 과정, 대안 후보, 자기수정, 레이블("최종 출력:" 등), 괄호 안 메모를
  절대 포함하지 마세요. 그 문장 외에는 한 글자도 출력하지 마세요.

질문: {question}
""")


def _extract_rewritten_question(raw_text):
    """LLM 재작성 응답에서 재작성된 질문 한 문장만 뽑아낸다.

    ※ 버그 수정: 프롬프트에 "한 문장만 출력"을 지시해도 일부 LLM(Upstage
    solar-pro 등)이 이를 어기고 자기수정·복수 후보·메타 코멘트("**최종 출력**:",
    "(※ ... 확인 필요)")를 함께 출력하는 경우가 있었다. 이걸 그대로 .strip()만
    해서 캐싱하면 그 잡음이 검색 질의 임베딩에 그대로 들어가 리트리버를 오염시킨다
    (실측: Q61 "직무발명보상금" 재작성이 통째로 캐싱되어 검색 실패로 이어짐).
    따옴표로 감싼 후보가 있으면 마지막 것(모델이 자기수정 후 마지막에 최종본을
    제시하는 경향)을 취하고, 없으면 레이블·메모성 줄을 건너뛰고 마지막
    비어있지 않은 줄을 취한다.
    """
    # 짧은 따옴표 키워드(예: 후행 메모의 "핵심 용어: '비과세', '한도'")가 문장으로
    # 오인식되지 않도록 일정 길이 이상인 따옴표 구간만 후보로 삼는다.
    quoted = [q.strip() for q in re.findall(r'["“]([^"”]+)["”]', raw_text) if len(q.strip()) > 15]
    if quoted:
        return quoted[-1]

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith(("※", "(", "**", "-", "*", "○", "▶")):
            continue
        return re.sub(r'^[^:：]{0,20}[:：]\s*', '', line).strip()
    return raw_text.strip()


def rewrite_query(question, llm_provider=config.LLM_PROVIDER):
    """[담당: 이희영] 질문을 문서 용어로 확장 재작성한다.

    handoff_ensemble.md "Q10 정밀 처방" ①: 질문에 제도명이 없어 임베딩이 정답
    페이지를 못 잇는 문제(예: Q10 표준세액공제, 정답이 유사도 33위)를
    특정 문항 하드코딩 없이 일반 규칙(LLM 재작성)으로 해결한다.
    """
    llm = get_llm(llm_provider)
    raw = (REWRITE_PROMPT | llm | StrOutputParser()).invoke({"question": question})
    return _extract_rewritten_question(raw)


def rrf_merge(doc_lists, rrf_k=60):
    """[담당: 이희영] 여러 검색 결과를 RRF(Reciprocal Rank Fusion)로 융합한다.

    점수 척도가 다른 검색 결과들을 점수가 아닌 '순위'로 공정하게 합친다:
    score(doc) = Σ 1/(rrf_k + rank). 양쪽에서 고루 상위인 문서가 최종 상위가
    되며, 한쪽 검색이 실패해도 다른 쪽이 보정한다. rrf_k=60은 관례적 표준값.
    """
    scores, docs_by_key = {}, {}
    for docs in doc_lists:
        for rank, d in enumerate(docs, start=1):
            key = (d.metadata.get("page"), d.metadata.get("chunk_id"))
            docs_by_key[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return [docs_by_key[k] for k in sorted(scores, key=scores.get, reverse=True)]


_RERANKER_CACHE = {}


def rerank_docs(question, docs, top_n=None, model_name=config.RERANKER_MODEL):
    """[담당: 이희영] cross-encoder로 (질문, 청크) 쌍을 직접 채점해 재정렬한다.

    이슈 #40 제안 ① (가이드라인 §12): 실패 대부분이 "정답이 후보 안에 있는데
    순위가 4위 밖"인 순위 밀림 유형 → 1차 검색(bi-encoder)은 후보를 넓게 뽑고,
    cross-encoder가 질문 맥락에서 각 청크를 정독 채점해 순위를 바로잡는다.
    모델은 최초 1회만 로드해 캐시한다 (재호출 비용 방지).
    """
    from sentence_transformers import CrossEncoder

    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    scores = _RERANKER_CACHE[model_name].predict(
        [(question, d.page_content) for d in docs]
    )
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    ranked = [docs[i] for i in order]
    return ranked[:top_n] if top_n else ranked


def format_docs(docs):
    """검색된 청크를 페이지 정보와 함께 하나의 문자열로 합친다."""
    return "\n\n".join(
        f"(p.{doc.metadata.get('page', '?')}) {doc.page_content}" for doc in docs
    )


def build_chain(vectorstore=config.VECTORSTORE,
                embedding=config.EMBEDDING_PROVIDER,
                llm_provider=config.LLM_PROVIDER,
                prompt_name=config.PROMPT_NAME,
                search_type=config.SEARCH_TYPE,
                top_k=config.TOP_K,
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP,
                embedding_model=None,
                dedup=config.DEDUP,
                fetch_k=config.DEDUP_FETCH_K):
    """RAG 체인과 retriever를 생성해 (chain, retriever)로 반환한다.

    embedding_model: 모델 ID 직접 지정 (예: intfloat/multilingual-e5-small).
                     None이면 config의 provider 기본 모델 사용.
    dedup/fetch_k: 페이지 단위 중복 제거, 기본값은 최종 확정 설정을 따른다
                  (이희영, dedup_effect.md/search_method_comparison.md).
                  main.py 챗봇도 이 기본값을 그대로 물려받는다.
    """
    vs = load_vectorstore(vectorstore, embedding, chunk_size, overlap,
                          embedding_model=embedding_model)
    retriever = get_retriever(vs, search_type, top_k, chunk_size, overlap,
                              dedup=dedup, fetch_k=fetch_k)
    llm = get_llm(llm_provider)
    prompt = PROMPTS[prompt_name]

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


LLM_CHOICES = ["gemini", "openai", "upstage", "claude", "qwen", "exaone"]
SEARCH_CHOICES = ["similarity", "mmr", "hybrid"]
EMBEDDING_CHOICES = ["huggingface", "openai", "gemini"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 질의응답")
    parser.add_argument("--question", required=True)
    parser.add_argument("--llm", choices=LLM_CHOICES, default=config.LLM_PROVIDER)
    parser.add_argument("--prompt", choices=list(PROMPTS), default=config.PROMPT_NAME)
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
