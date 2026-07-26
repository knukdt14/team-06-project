# 인수인계 2 — 평가 체계 강화 (Evaluation)

- **작성:** 이수홍 · 2026-07-27
- **관련:** [handover.md](handover.md)(이수민, 문서처리·검색) · [baseline_report.md](baseline_report.md)
- **배경:** 가이드라인 §2 "BERTScore만으로는 최적 모델을 찾기 어렵다" 대응

---

## 1. 지금까지 한 것

- ✅ **규칙 기반 채점기 `src/scoring.py` 신규** (LLM/API 불필요)
  - 핵심 숫자 정확도 · 조건 포함률 · 페이지 인용 정확도 · 거절 정확도
  - 오류 코드(E1~E11 중 자동 판정분) 태깅 → "E3 검색실패 몇 건" 식 분석 가능
- ✅ **재채점 도구 `src/rescore.py` 신규** — 이미 실행한 detail_*.csv를 API 없이 새 지표로 재채점
- ✅ **`evaluate.py`에 채점기 통합** — 앞으로 모든 실험에서 새 지표가 results.csv에 자동 기록
- ✅ **`references.csv` 확장** — 문항별 `key_numbers`(핵심 숫자) / `key_conditions`(필수 조건, `|`=동의어) / `answerable`(0=문서 밖 질문) 컬럼 추가

### 기존 결과 재채점 (측정 근거)

| 지표 | baseline (1000/200) | **cs500 (500/100)** |
|---|---|---|
| 핵심 숫자 정확도 | 0.769 | **0.821** |
| 조건 포함률 | 0.641 | **0.756** |
| 검색 Hit@3 | 0.154 | 0.385 |
| 오류 분포 | E3:11 · E7:2 · WR:3 | E3:8 · E6:2 · E7:2 · WR:2 |

**핵심 발견 — BERTScore가 못 잡던 치명적 오답 검출:**
cs500의 Q10은 표준세액공제 **13만원을 "7만원"이라고 답변**(엉뚱한 p.343 인용).
BERTScore로는 안 보였지만 E6(숫자 오류)로 자동 태깅됨. → 최적 설정 선택 시
**BERTScore보다 numeric_acc(숫자 정확도)를 우선 기준**으로 볼 것 (가이드라인 §2 중요도 순서).

## 2. 다음 사람이 할 일 (우선순위)

### ① 팀 논의: 검색 정답 페이지 기준 (제일 급함)
- [ ] E3(검색 실패)가 8~11건으로 나오지만 실제 답변 실패는 2~3건 — `references.csv`의 정답 page가 요약 페이지 기준이라 부풀려짐 (이수민 handover의 그 문제)
- [ ] "요약 페이지" vs "답 있는 아무 페이지" 결정 → 결정 전까지 E3 수치 해석 주의

### ② baseline 재현성 확정 (팀장)
- [ ] 현재 설정(cs500)으로 3회 반복 실행, 지표 변동폭 확인 (가이드라인 §1)
- [ ] ⚠️ 1회 = Gemini 13회 호출 → 무료 티어(20회/일)로는 하루 1회가 한계

### ③ 평가셋 확장 (팀장)
- [ ] 개발용 60~100문항 + **문서 밖 질문 10~20개** (가이드라인 §3)
- [ ] 문서 밖 질문은 `answerable=0`으로 넣으면 거절 정확도(abstention_acc)가 자동 채점됨 (현재 이 지표는 비어 있음)

### ④ 검색 설정 확정 후 (팀장)
- [ ] 프롬프트 비교(simple/basic/reasoning) → LLM 비교 → 최종 선정 (가이드라인 §14~15, §23)

## 3. 실행 방법 (명령어)

```bash
# 새 실험 — 기존과 동일, 채점 자동 포함 (Gemini 사용)
python src/evaluate.py --run-name <이름>

# 이미 실행한 결과 재채점 — API 불필요, 쿼터 절약
python src/rescore.py --detail eval/detail_baseline.csv --retrieval eval/retrieval_baseline.csv
python src/rescore.py --detail eval/detail_cs500.csv --retrieval eval/retrieval_cs500.csv

# 채점기 단위 테스트 (27개)
python -m pytest tests/test_scoring.py -q
```

results.csv에 추가되는 컬럼: `numeric_acc`, `numeric_perfect`, `condition_recall`,
`citation_acc`, `hit_rate`, `wrong_refusals`, `abstention_acc`, `error_counts`

## 4. 주의사항 (함정)

- ⚠️ **자동 태깅 커버리지** — E3·E6·E7·E9·E10·WR만 자동. **E1(추출 깨짐)·E2(청크 경계)·E8(환각)·E11(형식)은 수동 검토** 필요
- ⚠️ **숫자 매칭은 부분 문자열 기반** — "15%"가 "115%"에도 매칭될 수 있고, 부정문("700만원이 아님")도 정답 처리됨. 최종 발표 전 오답 문항은 눈으로 확인
- ⚠️ **표기 변형은 처리됨** — 1,500,000원=150만원, 100분의 90=90%, 8천만원=8000만원 (실제 Gemini 답변에서 관측된 것 반영)
- ⚠️ **WR(잘못 거절)은 부분 거절도 포함** — 일부만 답하고 "찾을 수 없다"가 섞인 답변도 WR로 태깅됨
- ⚠️ **E1 실제 사례 발견: PDF 물리 p.7 텍스트 추출이 완전히 깨짐**(폰트 인코딩) → 이수민 전처리 개선 대상
- ⚠️ **key_numbers에 쉼표 금지** — "1,000만원"이 아니라 "1000만원"으로 기입 (CSV 구분자 충돌, 채점은 쉼표 무시하므로 동일하게 동작)

## 5. 주요 파일

| 파일 | 담당 | 역할 |
|---|---|---|
| `src/scoring.py` | 이수홍 | 규칙 기반 채점 (신규) |
| `src/rescore.py` | 이수홍 | 기존 결과 재채점 (신규) |
| `src/evaluate.py` | 이수홍 | BERTScore + 규칙 채점 통합 |
| `eval/references.csv` | 공용 | 정답 + 채점 데이터 (key_numbers 등 3컬럼 추가) |
| `tests/test_scoring.py` | 이수홍 | 채점기 단위 테스트 27개 |
| `eval/detail_*_scored.csv` | - | baseline/cs500 재채점 결과 |
