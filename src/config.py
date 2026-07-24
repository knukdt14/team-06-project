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
CHUNK_SIZE = 1000          # 실험: 500 / 1000 / 1500
CHUNK_OVERLAP = 200        # 실험: 0 / 100 / 200
TOP_K = 3                  # 실험: 1 / 3 / 5

# 임베딩: "huggingface" | "gemini" | "openai"
# ※ Gemini 무료 티어는 임베딩 할당량이 작아 대량 문서 임베딩에 부적합 → 로컬 모델 기본
EMBEDDING_PROVIDER     = "huggingface"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
HF_EMBEDDING_MODEL     = "jhgan/ko-sroberta-multitask"   # 한국어 특화 비교용
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"        

# 벡터스토어: "chroma" | "faiss"
VECTORSTORE = "chroma"

# LLM: "gemini" | "upstage" | "openai"
LLM_PROVIDER      = "gemini"
GEMINI_LLM_MODEL  = "gemini-3.6-flash"
UPSTAGE_LLM_MODEL = "solar-pro"
OPENAI_LLM_MODEL  = "gpt-4o-mini"                  

# 검색 방식: "similarity" | "mmr"
SEARCH_TYPE = "similarity"


def vectorstore_path(vectorstore: str, embedding: str, chunk_size: int, overlap: int) -> Path:
    """실험 설정별로 벡터스토어를 따로 저장해 재사용한다."""
    name = f"{vectorstore}_{embedding}_cs{chunk_size}_ov{overlap}"
    return ARTIFACTS_DIR / name