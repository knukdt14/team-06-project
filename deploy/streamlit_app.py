"""Streamlit Community Cloud용 진입점.

실제 앱 로직은 프로젝트 루트의 app.py를 그대로 실행한다.
이 파일과 같은 폴더의 requirements.txt를 사용해 배포 의존성만 가볍게 유지한다.
로컬에서 검증한 최종 Chroma 스토어를 복원해 Cloud에서 재임베딩하지 않는다.
"""
import hashlib
from pathlib import Path
import runpy
import shutil


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
STORE_NAME = "chroma_huggingface_bge-m3_cs300_ov100"
STORE_ARCHIVE = DEPLOY_DIR / "vectorstore" / f"{STORE_NAME}.zip"
STORE_ARCHIVE_SHA256 = "2ECB7C283133CB7E4A3EE6A20FF77809CAA73CD3D91AF8C6795A16A3BB1B336E"
TARGET_STORE = PROJECT_ROOT / "artifacts" / STORE_NAME


def restore_prebuilt_vectorstore():
    """배포에 포함한 최종 Chroma 스토어를 artifacts 경로로 복원한다."""
    if TARGET_STORE.exists():
        return
    if not STORE_ARCHIVE.is_file():
        raise FileNotFoundError(f"배포용 벡터스토어가 없습니다: {STORE_ARCHIVE}")

    actual_hash = hashlib.sha256(STORE_ARCHIVE.read_bytes()).hexdigest().upper()
    if actual_hash != STORE_ARCHIVE_SHA256:
        raise RuntimeError(
            "배포용 벡터스토어 압축의 SHA-256이 일치하지 않습니다. "
            "파일이 손상됐을 수 있습니다."
        )

    TARGET_STORE.mkdir(parents=True)
    try:
        shutil.unpack_archive(str(STORE_ARCHIVE), str(TARGET_STORE))
    except Exception:
        shutil.rmtree(TARGET_STORE, ignore_errors=True)
        raise
    print(f"[deploy] 사전 구축 벡터스토어 복원 완료: {TARGET_STORE}")


restore_prebuilt_vectorstore()
ROOT_APP = PROJECT_ROOT / "app.py"
runpy.run_path(str(ROOT_APP), run_name="__main__")
