# 임베딩 모델 4종 검색 비교 — 상위 2개 선정 (가이드라인 §8)

- **작성:** 이희영 · 2026-07-27
- **목적:** 청킹·전처리 고정 상태에서 임베딩 모델만 변경해 검색 품질 비교,
  LLM 평가에 올릴 상위 2개 선정 (가이드라인 §8 "검색 상위 2개만 LLM 평가")
- **관련:** [page_criteria.md](page_criteria.md)(정답 기준) · [header_effect.md](header_effect.md)(전처리 기준) · 원본: `eval/retrieval_emb_*.csv`

---

## 1. 한눈에 (TL;DR)

- 비교 대상: 한국어 특화 1종 + 다국어 3종 (전부 로컬 HF 모델, API 비용 0원)
- ⭐ **상위 2개: `BAAI/bge-m3`, `jhgan/ko-sroberta-multitask`** → 팀장 LLM 평가로 전달
- 🔎 **핵심 발견 1:** bge-m3는 Hit@3에서 현재 모델과 동률(0.923)이지만 **MRR 0.923 vs 0.808로 확실히 우세** — 같은 문항을 맞혀도 정답 페이지를 더 높은 순위(대부분 1위)에 올림
- 🔎 **핵심 발견 2:** e5 계열(small·base)은 현재 모델보다 오히려 낮음 — 단, `query:`/`passage:` prefix 미적용 상태의 결과이므로 모델 자체의 한계로 단정할 수 없음

## 2. 실험 조건 (고정값)

| 항목 | 값 |
|---|---|
| chunk_size / overlap | 500 / 100 (청크 1,106개) |
| 전처리 | 머리말 제거 적용 (header_v1) |
| 벡터스토어 | Chroma (모델별 신규 빌드, 캐시 재사용 없음) |
| 검색 / top_k | similarity / 3 |
| 정답 기준 | 복수 정답 (page_criteria.md, `;` 구분) |
| 평가 도구 | `eval_retrieval.py` (LLM 미사용) |

## 3. 결과

| 모델 | 유형 | Hit@3 | MRR | 실패 문항 |
|---|---|---|---|---|
| **BAAI/bge-m3** | 다국어 | **0.923** | **0.923** | Q8 |
| jhgan/ko-sroberta-multitask (현재) | 한국어 특화 | **0.923** | 0.808 | Q10 |
| intfloat/multilingual-e5-base | 다국어 | 0.846 | 0.577 | Q1, Q9 |
| intfloat/multilingual-e5-small | 다국어 | 0.692 | 0.615 | Q1, Q4, Q6, Q8 |

- ko-sroberta 재측정값은 page_criteria.md의 재채점값(0.923/0.808)과 **일치** → 환경 재현 확인
- bge-m3는 13문항 중 12문항에서 정답 페이지를 **1위로** 검색 (RR=1.0), ko-sroberta는 9문항
- 참고: 모델 크기 — ko-sroberta 약 0.4GB · e5-small 약 0.5GB · e5-base 약 1.1GB · bge-m3 약 2.3GB
  (bge-m3는 임베딩 시간이 가장 오래 걸림 → 최종 선정 시 §23 "비용 대비" 관점에서 재검토 필요)

## 4. 실패 문항 분석

- **Q8 (부양가족 기본공제, 정답 p.4·10·113·117):** bge-m3·e5-small 공통 실패, 둘 다 `[58, 58, 58]` —
  p.58의 비슷한 청크 3개가 top 3을 독점. 정답 자체를 못 찾은 게 아니라 **중복 청크가 슬롯을 낭비**한 사례
- **Q10 (표준세액공제):** ko-sroberta만 실패 (`[387, 162, 344]` — 전부 무관 페이지). bge-m3는 p.188을 1위로 검색 →
  수민님 Phase 2의 마지막 타겟이던 Q10이 **임베딩 교체만으로 해결될 가능성**
- **Q1 (월세):** e5 계열만 실패 — 일상어 질문("얼마나 돌려받을 수 있나요")의 의미 매칭에서 밀림
- **중복 검색 관찰:** Q11은 e5·bge 모두 `[181, 181, 181]`, Q2도 같은 페이지 중복 다수 —
  top 3 슬롯이 같은 내용으로 채워지는 패턴이 모델 전반에서 반복됨 → **MMR(다양성) 실험 §10의 직접적 근거**

## 5. 결론 및 선정

**상위 2개: `BAAI/bge-m3` (1순위), `jhgan/ko-sroberta-multitask` (2순위)**

선정 근거:
- Hit@3는 두 모델 동률(0.923)이나 새 정답 기준에서는 Hit@3가 포화 상태 → 변별력 있는 **MRR 기준** bge-m3 우세 (0.923 vs 0.808)
- MRR이 높다는 것은 top_k를 줄여도 정답이 살아남는다는 뜻 → 이후 top_k 실험에서 유리
- e5 계열은 prefix 미적용 상태로도 순위가 낮아 제외 (prefix 적용 실험은 후속 검토 항목으로 남김)

## 6. 다음 단계 / 팀 영향

- **팀장님:** 상위 2개를 `evaluate.py --embedding-model <ID>`로 LLM 풀 평가 요청
  — 특히 **숫자 정확도**에서도 bge-m3 우위가 유지되는지 확인 (Gemini 쿼터 하루 20회, 일정 조율 필요)
- **수민님:** Q10은 bge-m3에서 해결됨. Q8의 중복 청크 문제(p.58 독점)는 청킹/중복 제거 관점에서도 참고
- **다음 실험(희영):** bge-m3 고정 후 top_k 실험(§9) → similarity/MMR/hybrid 비교(§10~11)
  — Q8·Q11의 중복 검색 사례가 MMR 효과 검증의 관찰 대상
- config.py의 `HF_EMBEDDING_MODEL` 변경(챔피언 교체)은 LLM 평가 결과 확인 후 진행

## 7. 재현 방법

```bash
# 전처리 변경이 있었던 경우 artifacts/ 삭제 후 실행
python src/eval_retrieval.py --run-name emb_kosroberta --chunk-size 500 --overlap 100
python src/eval_retrieval.py --run-name emb_e5small --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model intfloat/multilingual-e5-small
python src/eval_retrieval.py --run-name emb_e5base  --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model intfloat/multilingual-e5-base
python src/eval_retrieval.py --run-name emb_bgem3   --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3
```
