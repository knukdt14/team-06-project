# final66 검색 조합 비교 — rewrite+rerank 채택 (검색 파트 최종)

- **작성:** 이희영 · 2026-07-29
- **목적:** 66문항 정식 평가셋(PR #43)에서 검색 개선 도구들의 조합 비교, 최종 검색 조합 확정
  (이슈 #40 제안 ①·④ 검증 완료)
- **관련:** [markdown_loader_retrieval.md](markdown_loader_retrieval.md)(26문항 결과) ·
  이슈 #40(수민님 진단) · 원본: `eval/retrieval_f66_*.csv`

---

## 1. 한눈에 (TL;DR)

- ⭐ **최고 품질: ③ rewrite+rerank — Hit@3 0.850 / MRR 0.719** (기준선 0.783/0.667)
- 26문항 챔피언(② hybrid+rewrite)은 66문항에서 2위로 — "순위 밀림형 실패가 다수"라는
  이슈 #40 진단대로 **reranker가 결정타** (멀티페이지·헷갈림 유형 1.00 달성)
- ⚠️ **④ 풀조합(③+hybrid)은 ③보다 나쁨** — 도구 중첩이 항상 이득이 아님을 확인
  (§19 변수 간 상호작용: hybrid의 마크다운 표 청크가 reranker 채점과 간섭, Q18 재실패)
- 비용: reranker +9.8s/질문(CPU) → §23에 따라 2안 제시
  (최고 품질 ③ vs 비용 대비 ② — LLM 평가·팀 협의로 확정)
- 조건나열형(0.64)은 전 조합 무효 — 검색 방식 한계, 문서 구조(parent-child) 관점으로 이관

## 2. 실험 조건

고정: bge-m3 · cs500/ov100 · top_k 3 · 페이지 dedup · 복수 정답 기준 ·
final66 평가셋(답변가능 60 + 문서 밖 6). 변경: 로더 / rewrite / rerank 조합.

## 3. 전체 결과

| 조합 | Hit@3 | MRR | 실패 수 | 추가 비용 |
|---|---|---|---|---|
| ① 기준선 (pypdf + dedup) | 0.783 | 0.667 | 13 | — |
| ② hybrid 로더 + rewrite+RRF | 0.817 | 0.683 | 11 | 재작성 1회/질문 |
| **③ pypdf + rewrite+RRF + rerank** | **0.850** | **0.719** | **9** | 재작성 + **9.8s/질문** |
| ④ hybrid + rewrite+RRF + rerank | 0.833 | 0.686 | 10 | 재작성 + 10.8s/질문 |

## 4. 유형별 Hit@3 (qtype, PR #43 채점 기능)

| 유형 (문항 수) | ① 기준선 | ② hyb+rrf | ③ rrf+rerank | ④ 풀조합 |
|---|---|---|---|---|
| 단순사실형 (12) | 0.92 | 0.83 | 0.92 | 0.92 |
| 계산형 (11) | 0.91 | 0.91 | 0.91 | 0.91 |
| 멀티페이지 (8) | 0.75 | 0.88 | **1.00** | 0.88 |
| 헷갈림 (7) | 0.71 | 0.86 | **1.00** | **1.00** |
| 일상어 (11) | 0.73 | **0.82** | 0.73 | 0.73 |
| 조건나열형 (11) | 0.64 | 0.64 | 0.64 | 0.64 |

## 5. 해석

1. **reranker의 주 무대는 멀티페이지·헷갈림** — 정답이 후보 안에 있는데 순위가 밀리는
   유형에서 완승(각각 0.75→1.00, 0.71→1.00). 이슈 #40 진단의 정확한 검증.
2. **reranker의 부작용** — Q1·Q14·Q19 강등(일상어 계열): rewrite가 끌어올린 정답을
   cross-encoder가 다시 내리는 사례. 일상어 유형에서 ②(0.82) 대비 ③(0.73)이 낮은 원인.
   순이득은 크게 양수(+4문항)이나, 유형별로는 손해 보는 곳이 있음.
3. **④ < ③ (조합 간섭)** — hybrid의 마크다운 표 청크가 reranker 채점에서 원문 청크와
   다르게 평가되며 Q18 재실패 등 발생. "좋은 도구 합 ≠ 더 좋은 결과"의 실증 (§19).
4. **조건나열형 0.64는 요지부동** — 근거가 여러 절에 흩어진 유형으로 추정, 검색 알고리즘
   교체로는 해결 불가. 수민님 parent-child retrieval(PR #41, 멀티페이지 커버 개선)과
   연계 진단 필요 → 이관.

## 6. 결론 (§23: 2안 제시)

| | 조합 | 근거 |
|---|---|---|
| **최고 품질** | ③ rewrite+rerank | Hit@3·MRR 최고. 단 +9.8s/질문(CPU) |
| **비용 대비 최적** | ② hybrid+rewrite | rerank 없이 0.817, 응답시간 부담 없음 |

최종 확정은 LLM 풀 평가(숫자 정확도·전체 응답시간)와 서비스 관점 협의로 —
GPU 사용 시 rerank 시간이 크게 줄 수 있어 환경 전제도 함께 검토.

## 7. 다음 단계 / 팀 영향

- **팀장님:** ②·③ 두 조합 LLM 풀 평가 요청 (검색 개선이 숫자 정확도로 이어지는지 +
  rerank 지연 포함 총 응답시간)
- **수민님:** 조건나열형 공통 실패(Q53·54·56 등) 이관 — parent-child와 연계 진단.
  이슈 #40 제안 ①·③·④ 검증 완료 보고 (② hybrid 재구성은 ④ 간섭 결과로 보류 제안)
- 소형 reranker(`BAAI/bge-reranker-base`) 속도-성능 비교는 ③ 채택 시 후속 검토
- 과적합 방지: 최종 확정 후 테스트셋(별도 보관분) 1회 검증 필요 (§3·§23)

## 8. 재현 방법

```bash
python src/eval_retrieval.py --run-name f66_dedup          --chunk-size 500 --overlap 100 --top-k 3
python src/eval_retrieval.py --run-name f66_hyb_rrf        --chunk-size 500 --overlap 100 --top-k 3 --loader hybrid --rewrite
python src/eval_retrieval.py --run-name f66_rrf_rerank     --chunk-size 500 --overlap 100 --top-k 3 --rewrite --rerank
python src/eval_retrieval.py --run-name f66_hyb_rrf_rerank --chunk-size 500 --overlap 100 --top-k 3 --loader hybrid --rewrite --rerank
```
