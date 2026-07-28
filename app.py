"""
연말정산 Q&A 데모 (Streamlit)

실행:
    python -m streamlit run app.py

rag_chain.py의 함수를 그대로 재사용한다 (검색 로직 수정 없음).
- 빠른 답변: dedup 검색 → LLM
- 정밀 답변: 넓게 검색 → reranker 재정렬 → LLM
"""
import html
import os
import sys
from time import perf_counter

sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()                       # .env의 UPSTAGE_API_KEY 등 로드

import streamlit as st
import config

st.set_page_config(page_title="연말정산 Q&A", page_icon="💰", layout="centered")

# ────────────────────────── 디자인 (CSS) ──────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #ecfeff 0%, #f0fdf4 45%, #eef2ff 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 720px; }

.hero-title { font-size: 3rem; font-weight: 800; color: #0f172a;
              letter-spacing: -0.03em; margin-bottom: .15rem; }
.hero-sub   { color: #52606d; font-size: 1.02rem; margin-bottom: 1.6rem; }

.answer-card {
    background: #ffffff; border-radius: 16px; padding: 1.4rem 1.6rem;
    box-shadow: 0 10px 34px rgba(13,148,136,.10); line-height: 1.75;
    color: #1e293b; white-space: pre-wrap; font-size: 1.03rem;
}
.src-chip {
    display: inline-block; background: #ccfbf1; color: #0f766e;
    border-radius: 999px; padding: .18rem .8rem; margin: .25rem .25rem 0 0;
    font-size: .84rem; font-weight: 600;
}
.evidence-card {
    background: #f8fafc; border-left: 4px solid #14b8a6;
    border-radius: 8px; padding: .8rem 1rem; margin: .55rem 0 1rem;
    color: #334155; white-space: pre-wrap; line-height: 1.65;
}
.stButton>button { border-radius: 10px; font-weight: 600; font-size: 1.06rem; }
.stTextInput input { font-size: 1.06rem; }
div[role="radiogroup"] label p, div[role="radiogroup"] label { font-size: 1.05rem; }
[data-testid="stWidgetLabel"] p { font-size: 1.03rem; }
[data-testid="stCaptionContainer"] p { font-size: .95rem; }
footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────── 실행 전 상태 점검 ──────────────────────────
def get_runtime_status():
    """API 키·PDF·벡터스토어 준비 여부를 실제 최종 설정 기준으로 확인한다."""
    api_key_names = {
        "upstage": "UPSTAGE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    api_key_name = api_key_names.get(config.LLM_PROVIDER)
    api_ready = api_key_name is None or bool(os.getenv(api_key_name))

    embedding_models = {
        "huggingface": config.HF_EMBEDDING_MODEL,
        "openai": config.OPENAI_EMBEDDING_MODEL,
        "gemini": config.GEMINI_EMBEDDING_MODEL,
    }
    embedding_model = embedding_models.get(config.EMBEDDING_PROVIDER, "")
    vectorstore_path = config.vectorstore_path(
        config.VECTORSTORE,
        config.EMBEDDING_PROVIDER,
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
        embedding_model,
        config.PDF_LOADER,
    )

    return {
        "api_ready": api_ready,
        "api_label": f"{config.LLM_PROVIDER} API",
        "api_key_name": api_key_name,
        "pdf_ready": config.PDF_PATH.is_file(),
        "vectorstore_ready": vectorstore_path.exists(),
        "vectorstore_path": vectorstore_path,
    }


def render_runtime_status(status):
    """발표 전에 필요한 실행 자원을 한눈에 확인할 수 있게 표시한다."""
    with st.expander("🩺 실행 준비 상태", expanded=not all([
        status["api_ready"],
        status["pdf_ready"],
        status["vectorstore_ready"],
    ])):
        api_col, pdf_col, vector_col = st.columns(3)
        api_col.metric(
            status["api_label"],
            "✅ 준비됨" if status["api_ready"] else "❌ 키 없음",
        )
        pdf_col.metric(
            "신고안내 PDF",
            "✅ 준비됨" if status["pdf_ready"] else "❌ 파일 없음",
        )
        vector_col.metric(
            "벡터스토어",
            "✅ 준비됨" if status["vectorstore_ready"] else "⚠️ 미구축",
        )

        if not status["api_ready"]:
            st.error(f".env에 {status['api_key_name']}를 설정해 주세요.")
        if not status["pdf_ready"]:
            st.error(f"PDF를 찾을 수 없습니다: {config.PDF_PATH}")
        if not status["vectorstore_ready"]:
            st.warning(
                "첫 질문 때 벡터스토어를 자동으로 구축하므로 오래 걸릴 수 있습니다. "
                "발표 전 `python src/build_vectorstore.py`를 한 번 실행해 주세요."
            )
        st.caption(f"벡터스토어 경로: {status['vectorstore_path']}")


# ────────────────────────── 엔진 로딩 (1회 캐시) ──────────────────────────
@st.cache_resource(show_spinner="🔧 모델 로딩 중… (처음 한 번만, 20초쯤 걸려요)")
def load_engine():
    from build_vectorstore import load_vectorstore
    from rag_chain import get_retriever, get_llm, PROMPTS

    vs = load_vectorstore()
    retriever = get_retriever(vs, config.SEARCH_TYPE, config.TOP_K,
                              config.CHUNK_SIZE, config.CHUNK_OVERLAP,
                              dedup=config.DEDUP, fetch_k=config.DEDUP_FETCH_K)
    llm = get_llm(config.LLM_PROVIDER)
    prompt = PROMPTS[config.PROMPT_NAME]
    return vs, retriever, llm, prompt, config


def answer_question(question, precise):
    """답변·근거 문서와 엔진/검색/생성 단계별 소요시간을 반환한다."""
    from rag_chain import rerank_docs, format_docs
    from langchain_core.output_parsers import StrOutputParser

    engine_started = perf_counter()
    vs, retriever, llm, prompt, config = load_engine()
    engine_sec = perf_counter() - engine_started

    search_started = perf_counter()
    if precise:                                       # 정밀: 넓게 검색 → 재정렬 → top-k
        candidates = vs.similarity_search(question, k=config.DEDUP_FETCH_K)
        docs = rerank_docs(question, candidates, top_n=config.TOP_K)
    else:                                             # 빠른: dedup 검색
        docs = retriever.invoke(question)
    search_sec = perf_counter() - search_started

    generation_started = perf_counter()
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(docs), "question": question}
    )
    generation_sec = perf_counter() - generation_started
    timings = {
        "engine_sec": engine_sec,
        "search_sec": search_sec,
        "generation_sec": generation_sec,
    }
    return answer, docs, timings


@st.cache_data(show_spinner=False)
def render_pdf_page(page_index):
    """PDF의 특정 페이지(0-indexed)를 PNG 이미지로 렌더링한다."""
    import fitz
    import config
    with fitz.open(str(config.PDF_PATH)) as doc:
        pix = doc[page_index].get_pixmap(dpi=130)
        return pix.tobytes("png")


FRONT_MATTER = 18   # 표지·목차 등 인쇄번호 없는 앞부분 (load_pdf.PRINTED_PAGE_OFFSET와 동일)

# PDF 내장 글꼴의 ①~⑪이 pypdf에서 잘못 해석되는 경우를 화면에서만 복원한다.
# 검색·임베딩에 사용하는 Document 원문은 수정하지 않는다.
PDF_GLYPH_DISPLAY_MAP = {
    "쇮쇱": "①",
    "쇮쇲": "②",
    "쇮쇳": "③",
    "쇮쇴": "④",
    "쇮쇵": "⑤",
    "쇮쇶": "⑥",
    "쇮쇷": "⑦",
    "쇮쇸": "⑧",
    "쇮쇹": "⑨",
    "쇮쇺": "⑩",
    "쇮쇻": "⑪",
}


def normalize_pdf_glyphs(text):
    """깨진 PDF 번호 기호를 Streamlit 표시용 문자열에서만 복원한다."""
    for broken, restored in PDF_GLYPH_DISPLAY_MAP.items():
        text = text.replace(broken, restored)
    return text


def page_label(page_index):
    """metadata page(0-indexed) → 문서에 실제로 찍힌 인쇄 페이지 번호 라벨."""
    printed = page_index + 1 - FRONT_MATTER
    return f"p.{printed}" if printed >= 1 else f"앞부분 {page_index + 1}번째 장"


# ────────────────────────── 헤더 ──────────────────────────
st.markdown('<div class="hero-title">💰 연말정산 Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">「2025년 연말정산 신고안내」를 학습한 AI에게 물어보세요.</div>',
            unsafe_allow_html=True)

runtime_status = get_runtime_status()
render_runtime_status(runtime_status)

# ────────────────────────── 입력 ──────────────────────────
if "q" not in st.session_state:
    st.session_state.q = ""

mode = st.radio(
    "답변 모드",
    ["⚡ 빠른 답변", "🎯 정밀 답변"],
    horizontal=True,
    captions=["즉시 응답", "재정렬로 더 정확 (몇 초 더)"],
)
precise = mode.startswith("🎯")

st.caption("예시 질문 — 눌러보세요")
examples = [
    "월세 살면 얼마나 돌려받나요?",
    "8세 이상 자녀 2명이면 세액공제 얼마?",
    "산후조리원 비용도 공제되나요?",
]
for col, ex in zip(st.columns(len(examples)), examples):
    if col.button(ex, use_container_width=True):
        st.session_state.q = ex

st.text_input("질문", key="q", label_visibility="collapsed",
              placeholder="예: 신용카드 소득공제는 어떻게 계산하나요?")

critical_resources_ready = runtime_status["api_ready"] and runtime_status["pdf_ready"]
go = st.button(
    "질문하기",
    type="primary",
    use_container_width=True,
    disabled=not critical_resources_ready,
)

# ────────────────────────── 실행 ──────────────────────────
if go and st.session_state.q.strip():
    total_started = perf_counter()
    with st.spinner("🔎 문서 검색 + 답변 생성 중…"):
        try:
            answer, docs, timings = answer_question(st.session_state.q.strip(), precise)
        except Exception as e:
            st.error(f"오류가 발생했어요: {e}")
            st.stop()
    total_sec = perf_counter() - total_started

    st.markdown("#### 💬 답변")
    display_answer = normalize_pdf_glyphs(answer)
    st.markdown(
        f'<div class="answer-card">{html.escape(display_answer)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### ⏱️ 응답시간")
    total_col, search_col, generation_col = st.columns(3)
    total_col.metric("전체", f"{total_sec:.2f}초")
    search_col.metric("문서 검색", f"{timings['search_sec']:.2f}초")
    generation_col.metric("답변 생성", f"{timings['generation_sec']:.2f}초")
    if timings["engine_sec"] >= 0.1:
        st.caption(
            f"첫 실행 모델·벡터스토어 준비 시간 {timings['engine_sec']:.2f}초가 "
            "전체 시간에 포함됐습니다."
        )

    st.markdown("#### 📄 근거 페이지")
    pages = list(dict.fromkeys(
        d.metadata.get("page") for d in docs
        if isinstance(d.metadata.get("page"), int)
    ))   # 0-indexed, 중복 제거
    chips = "".join(f'<span class="src-chip">{page_label(p)}</span>' for p in pages)   # 인쇄 페이지 번호
    st.markdown(chips, unsafe_allow_html=True)

    with st.expander("🔍 검색 근거 문장 보기", expanded=True):
        for rank, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            label = page_label(page) if isinstance(page, int) else "페이지 정보 없음"
            st.markdown(f"**검색 {rank}위 · {label}**")
            evidence_text = normalize_pdf_glyphs(doc.page_content.strip())
            st.markdown(
                f'<div class="evidence-card">{html.escape(evidence_text)}</div>',
                unsafe_allow_html=True,
            )

    with st.expander("📄 실제 PDF 페이지 보기"):
        for p in pages:
            st.image(render_pdf_page(p), caption=f"연말정산 신고안내 {page_label(p)}",
                     use_container_width=True)
