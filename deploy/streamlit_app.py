"""Streamlit Community Cloud용 진입점.

실제 앱 로직은 프로젝트 루트의 app.py를 그대로 실행한다.
이 파일과 같은 폴더의 requirements.txt를 사용해 배포 의존성만 가볍게 유지한다.
"""
from pathlib import Path
import runpy


ROOT_APP = Path(__file__).resolve().parent.parent / "app.py"
runpy.run_path(str(ROOT_APP), run_name="__main__")
