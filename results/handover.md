# 인수인계 — 문서처리·검색 (Document Processing & Retrieval)

- **작성:** 이수민 · 2026-07-26
- **관련:** [baseline_report.md](baseline_report.md) · [detail_baseline.md](detail_baseline.md)

---

## 1. 지금까지 한 것

- ✅ **베이스라인 측정** + **검색 채점기 `src/eval_retrieval.py` 신규** (LLM 없이 Hit@k/MRR 측정)
- ✅ **1차 개선: `chunk_size` 1000 → 500 채택** (`config.py`)

| 지표 | baseline (1000/200) | **cs500 (500/100)** |
|---|---|---|
| 검색 Hit@3 | 0.154 | **0.385** |
| BERTScore F1 | 0.6961 | **0.7423** |

→ 검색·답변 **둘 다 개선 확인.**

## 2. 다음 사람이 할 일 (우선순위)

### ① Phase 2: 문서처리 구조 개선 (제일 중요)
아직 **진짜 답변 실패 3문항**이 남음 → 이걸 잡는 게 목표:
- **Q1** 월세 금액 / **Q9** 70세 경로우대 / **Q10** 표준세액공제 (모두 "문서에서 찾을 수 없습니다")

할 것 (하나씩 PR로, 매번 점수 측정):
- [ ] **섹션 단위 청킹** — 요약 페이지(4·5·6)에 여러 공제가 뭉쳐 있어 검색이 희석됨. 항목/제목 단위로 자르기
- [ ] **표 전처리 / PDF→마크다운** — 세율표 등 표가 납작하게 뭉개짐 (`load_pdf.py`)
- [ ] (선택) 멀티모달 — 위로도 안 되면

### ② 팀 논의 필요
- [ ] **검색 정답 기준 정렬** — 현재 `references.csv`의 정답 page가 "앞쪽 요약 페이지" 기준이라 Hit@3이 실제 성능을 과소평가함 (10/13은 상세 페이지에서 답을 찾아 정상 답변). "요약 페이지" vs "답 있는 아무 페이지" 중 뭘 정답으로 볼지 이희영/팀장과 상의

### ③ 추가 탐색 (여유 되면)
- [ ] overlap, top_k 미세조정 (cs500 기준)
- [ ] 임베딩 모델 비교는 **이희영 담당**과 협의

## 3. 실행 방법 (명령어)

실행 전 (Windows): `$env:PYTHONIOENCODING="utf-8"` · `.env`에 `GOOGLE_API_KEY=<Gemini키>`

```bash
# 검색 채점 (API 불필요, 빠름 — 반복 실험용)
python src/eval_retrieval.py --run-name <이름> --chunk-size 500 --overlap 100

# 최종 평가 (Gemini 사용 — BERTScore)
python src/evaluate.py --run-name <이름> --chunk-size 500 --overlap 100
```
결과는 `eval/results.csv`에 누적됨.

## 4. 주의사항 (함정)

- ⚠️ **Gemini 무료 티어 20회/일** — baseline + 실험 몇 개면 소진(429). 하루 예산 관리하거나 여분/유료 키 필요
- ⚠️ **페이지 정합** — `references.csv`의 page는 1-indexed, 청크 `metadata['page']`는 0-indexed → 채점 시 `+1` 보정 (eval_retrieval.py에 반영됨)
- ⚠️ **ragas는 안 씀** — 버전 비호환(0.4.3 ↔ langchain-community). Phase 0엔 불필요. 필요 시 팀장(evaluate.py 담당)이 처리
- ⚠️ **역할 경계** — `evaluate.py`(팀장), 검색 알고리즘(이희영)은 수정 X. 검색 채점은 별도 `eval_retrieval.py`

## 5. 주요 파일

| 파일 | 담당 | 역할 |
|---|---|---|
| `src/load_pdf.py` | 이수민 | PDF 로드·청킹 (**Phase 2 주 무대**) |
| `src/build_vectorstore.py` | 이수민 | 임베딩·벡터스토어 |
| `src/eval_retrieval.py` | 이수민 | 검색 채점 (신규) |
| `src/config.py` | 공용 | 실험 파라미터 (chunk_size=500 채택) |
| `src/evaluate.py` | 팀장 | BERTScore 평가 (수정 X) |
