"""
프로젝트 공통 설정.

실험 파라미터를 한 곳에서 관리한다.
(chunk_size, top_k 등을 바꿔가며 성능 비교 실험을 할 때 이 파일 또는 CLI 인자를 사용)

""" 
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_DIR / "data"
EVAL_DIR      = PROJECT_DIR / "eval"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

PDF_PATH = DATA_DIR / "2025년 원천징수의무자를 위한 연말정산 신고안내.pdf"

# ── 실험 파라미터 (기본값) ──────────────
CHUNK_SIZE = 300           # 재튜닝: bge-m3+66문항 기준 500→300 (Hit@3 0.783→0.883, 공짜). 구: 1000→500(ko-sroberta+13문항)
CHUNK_OVERLAP = 100        # 실험: 0 / 50 / 100 / 200
TOP_K = 3                  # 실험: 1 / 3 / 5

# PDF 로더: "pypdf"(일반 텍스트) | "markdown"(pymupdf4llm 전체 변환) | "hybrid"(표 페이지만 마크다운)
# ※ 실험: 마크다운은 표가 깨지지 않아 계산형·조건나열형 문항의 검색 개선을 노린다.
#   단 전체 변환(markdown)은 Q18·Q14를 고치는 대신 Q6·Q10을 깨뜨리고 MRR을
#   0.826→0.638로 떨어뜨렸다(표 아닌 본문까지 서식이 섞임) → 순수익 없음.
#   hybrid는 표가 검출된 페이지만 교체해 이 부작용을 피한다(load_pdf.load_documents_hybrid).
PDF_LOADER = "pypdf"

# 임베딩: "huggingface" | "gemini" | "openai"
# ※ Gemini 무료 티어는 임베딩 할당량이 작아 대량 문서 임베딩에 부적합 → 로컬 모델 기본
# ※ 최종 확정(§8, embedding_comparison.md): bge-m3가 ko-sroberta보다 MRR 우세(0.923 vs 0.808)
#   + LLM 평가로도 재확인(final_model_selection.md) → 기본값 교체
EMBEDDING_PROVIDER     = "huggingface"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
HF_EMBEDDING_MODEL     = "BAAI/bge-m3"                   # 최종 채택 (구: jhgan/ko-sroberta-multitask)
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# 벡터스토어: "chroma" | "faiss"
VECTORSTORE = "chroma"

# LLM: "gemini" | "upstage" | "openai"
# ※ 최종 확정(final_model_selection.md): Upstage가 4회 반복에서 OpenAI 대비 일관 우세
#   + Gemini 대비 응답시간 약 3배 빠름 → 최고 품질과 가성비가 동시에 수렴
LLM_PROVIDER      = "upstage"                            # 최종 채택 (구: gemini)
GEMINI_LLM_MODEL  = "gemini-3.6-flash"
UPSTAGE_LLM_MODEL = "solar-pro"
OPENAI_LLM_MODEL  = "gpt-5.4-mini"

# 검색 방식: "similarity" | "mmr" | "hybrid"
# ※ 최종 확정(search_method_comparison.md): similarity+dedup이 MMR·hybrid 도전자 전패시킴
SEARCH_TYPE = "similarity"

# 페이지 dedup (이희영, dedup_effect.md/search_method_comparison.md 최종 확정)
# fetch_k개를 1차 검색 후 서로 다른 페이지 top_k개만 남긴다. Hit@3 0.923→1.000(13/13).
DEDUP          = True
DEDUP_FETCH_K  = 15

# hybrid 검색: [BM25, Dense(벡터)] 가중치, 합=1 — 실험: 0.2/0.5/0.8 (검색 방식 3파전에서 탈락)
HYBRID_WEIGHTS = [0.5, 0.5]

# reranker (이슈 #40 제안 ①, 가이드라인 §12): 1차 검색 후보를 cross-encoder로 재정렬.
# bge-m3와 같은 BAAI 계열의 다국어 reranker, 로컬 실행(API 불필요, 약 2GB).
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 프롬프트: "basic" | "simple" | "reasoning" (rag_chain.py PROMPTS)
# ※ 최종 확정(prompt_experiment.md): reasoning이 basic 대비 numeric_acc는
#   완전히 동일한 오답 패턴(Q6/Q10/Q13 동일)이면서 BERTScore 개선(+0.03)·
#   응답시간 단축(-25%, 3회 반복 모두 일관). simple은 numeric_acc가 더
#   높지만(Q6/Q13 개선) BERTScore 하락 + 응답시간 +65% + "문서에 없으면
#   모른다" 지시 부재로 인한 할루시네이션 리스크 우려로 기각.
PROMPT_NAME = "reasoning"                                # 최종 채택 (구: basic)


def vectorstore_path(vectorstore: str, embedding: str,
                     chunk_size: int, overlap: int,
                     embedding_model: str = "", loader: str = "pypdf") -> Path:
    """실험 설정별로 벡터스토어를 따로 저장해 재사용한다."""
    model_tag = f"_{embedding_model.split('/')[-1]}" if embedding_model else ""
    # pypdf는 태그 없음 → 기존 경로 유지(하위호환). 로더가 다르면 스토어도 분리돼야 한다.
    loader_tag = {"markdown": "_md", "hybrid": "_hyb"}.get(loader, "")
    name = f"{vectorstore}_{embedding}{model_tag}_cs{chunk_size}_ov{overlap}{loader_tag}"
    return ARTIFACTS_DIR / name