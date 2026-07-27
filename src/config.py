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
CHUNK_SIZE = 500           # 실험 결과 채택: 1000→500 (Hit@3 0.15→0.39, BERTScore 0.70→0.74). 후보: 300 / 500 / 1000
CHUNK_OVERLAP = 100        # 실험: 0 / 50 / 100 / 200
TOP_K = 3                  # 실험: 1 / 3 / 5

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


def vectorstore_path(vectorstore: str, embedding: str, 
                     chunk_size: int, overlap: int,
                     embedding_model: str = "") -> Path:
    """실험 설정별로 벡터스토어를 따로 저장해 재사용한다."""
    model_tag = f"_{embedding_model.split('/')[-1]}" if embedding_model else ""
    name = f"{vectorstore}_{embedding}{model_tag}_cs{chunk_size}_ov{overlap}"
    return ARTIFACTS_DIR / name