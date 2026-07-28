# final66 검색 조합 비교 — cs300 + rerank(단독) 최종 채택 (검색 파트 최종)

- **작성:** 이희영 · 2026-07-29 (cs300 재검증 + rewrite 기여도 분해 반영)
- **목적:** 66문항 정식 평가셋(PR #43)에서 검색 개선 도구 조합 비교, 최종 검색 조합 확정
  (이슈 #40 제안 ①·④ 검증 완료 + chunk_size 재검증 + rewrite 단독 기여도 분해)
- **관련:** [markdown_loader_retrieval.md](markdown_loader_retrieval.md)(26문항 결과) ·
  이슈 #40(수민님 진단) · 원본: `eval/retrieval_f66_*.csv`

---

## 1. 한눈에 (TL;DR)

- ⭐ **최종 채택: cs300 + rerank (rewrite 제외) — Hit@3 0.917 / MRR 0.794** — 지금까지의
  모든 조합 중 최고, 게다가 rewrite를 뺀 더 단순한 구성 (팀원 제안으로 발견)
- **rewrite는 이 조합에서 기여 0** — rerank만으로 못 잡던 문항을 rewrite가 추가로 잡아준
  사례가 0건. 오히려 rewrite가 후보 구성을 바꿔 rerank 판단을 미세 방해(Q35)한 흔적 →
  **API 호출·응답시간 부담만 있고 이득 없음, 폐기 권고**
- **chunk_size 재검증이 핵심 변수였다** — cs300 기준선(dedup만, 0.883)만으로도 cs500
  최고 조합(0.850, 구 rewrite+rerank)을 넘어섬. 기존 cs500 채택은 구 조건(임베딩·평가셋·
  정답 기준) 기준이라 현재 조건에서 재검증이 유효했음
- rerank 단독은 비용도 가장 낮음 — rewrite 포함 5.8s → **rewrite 제외 4.2s/질문**
  (재작성 API 호출·대기 시간이 사라진 만큼)
- ⚠️ hybrid 로더는 cs300에서 역효과(0.883→0.783) — **폐기 권고**
- 조건나열형은 cs300+rewrite+rerank(0.82)가 cs300+rerank단독보다 근소 우위 —
  유일하게 남은 트레이드오프, §4·§6 참고

## 2. 실험 조건

고정: bge-m3 · overlap 100 · top_k 3 · 페이지 dedup · 복수 정답 기준 · final66 평가셋(답변가능 60 + 문서 밖 6).
변경: **chunk_size(500/300) × 로더(pypdf/hybrid) × rewrite × rerank**.

## 3. 전체 결과 (9조합)

| chunk | 조합 | Hit@3 | MRR | 실패 | rerank 시간 |
|---|---|---|---|---|---|
| 500 | 기준선 (dedup) | 0.783 | 0.667 | 13 | — |
| 500 | hybrid + rewrite | 0.817 | 0.683 | 11 | — |
| 500 | rewrite + rerank | 0.850 | 0.719 | 9 | 9.8s |
| 500 | 풀조합 | 0.833 | 0.686 | 10 | 10.8s |
| 300 | 기준선 (dedup) | 0.883 | 0.680 | 7 | — |
| 300 | hybrid + rewrite | 0.783 | 0.619 | 13 | — |
| **300** | **⭐ rerank 단독 (rewrite 없음)** | **0.917** | **0.794** | **5** | **4.2s** |
| 300 | rewrite + rerank | 0.900 | 0.778 | 6 | 5.8s |
| 300 | 풀조합(hybrid+rewrite+rerank) | 0.850 | 0.711 | 9 | 6.5s |

### rewrite 기여도 분해 (cs300, dedup→rerank단독→rewrite+rerank)

| | Hit@3 | 새로 잡은 문항 | 새로 깨뜨린 문항 |
|---|---|---|---|
| dedup → **rerank 단독** | 0.883 → **0.917** | Q8, Q56 | 없음 |
| rerank 단독 → rewrite+rerank | 0.917 → 0.900 | **0건** | Q35 |

→ **rewrite가 rerank 단독으로는 못 잡는 문항을 추가로 해결한 사례가 0건.**
오히려 확장 질문이 후보 구성을 바꿔 rerank 판단을 미세 방해(Q35 신규 실패)한 것으로 추정.

## 4. 유형별 Hit@3 (qtype, PR #43 채점 기능)

| 유형 (문항 수) | cs500 기준선 | cs500 rrf+rerank | cs300 기준선 | **cs300 rerank단독** | cs300 rrf+rerank |
|---|---|---|---|---|---|
| 단순사실형 (12) | 0.92 | 0.92 | 1.00 | **1.00** | 1.00 |
| 계산형 (11) | 0.91 | 0.91 | 1.00 | **1.00** | 1.00 |
| 멀티페이지 (8) | 0.75 | 1.00 | 0.75 | 0.75 | 0.75 |
| 헷갈림 (7) | 0.71 | 1.00 | 1.00 | **1.00** | 1.00 |
| 일상어 (11) | 0.73 | 0.73 | 0.91 | **0.91** | 0.82 |
| 조건나열형 (11) | 0.64 | 0.64 | 0.64 | 0.82 | **0.82** |

※ 조건나열형·일상어는 rerank단독과 rewrite+rerank가 동률이거나 rerank단독이 더 우세 —
rewrite를 빼도 유형별로 손해 보는 곳이 없음 (일상어는 오히려 단독이 더 높음: 0.91 vs 0.82).

## 5. 해석

1. **chunk_size가 이번 라운드의 최대 변수였다** — cs300 기준선만으로도 cs500의 최고 조합을 넘어섬.
   원인 추정: 작은 청크일수록 한 청크가 한 주제에 집중돼 bge-m3 임베딩의 의미 매칭이 정밀해짐
   (일상어 0.73→0.91, 헷갈림 0.71→1.00). 기존 cs500 채택은 다른 임베딩·평가셋·정답 기준
   시절의 결론이라 현재 조건에서 재검증이 유효했음.
2. **rewrite는 이 조합에서 불필요 (신규 발견)** — 팀원 제안으로 "rewrite 없이 rerank만"을
   시험한 결과, rerank 단독(0.917)이 rewrite+rerank(0.900)보다 오히려 높음. §3 기여도 분해
   표대로 rewrite가 추가로 해결한 문항은 0건, 반대로 후보 구성 변화로 Q35를 새로 깨뜨림.
   과거(26문항, hybrid 로더 시절) rewrite가 Q14·Q19를 잡던 효과는 cs300+rerank 조합에서는
   더 이상 필요 없어진 것으로 판단 — rerank가 순위 밀림형뿐 아니라 그 유형들도 흡수.
3. **조건나열형 돌파구** — cs300(잘게 쪼개진 근거) + rerank(정밀 재정렬) 조합에서만 0.82 도달
   (rewrite 유무와 무관하게 동일). 청킹만으로도, rerank만으로도 안 풀리던 유형이 두 조건이
   겹쳐야 풀림 — §19 변수 간 상호작용의 긍정적 사례.
4. **hybrid 로더는 cs300에서 명백히 손해** (0.883→0.783, 13개 실패로 급증) — cs500에서도
   이미 열세였는데(0.817 vs 0.850) cs300에서는 방향이 완전히 뒤집힘. **더 이상 채택 후보 아님.**
5. **rerank 부작용은 cs300에서도 존재하나 축소** — 멀티페이지는 cs500+rerank(1.00) 대비
   cs300+rerank(0.75)로 낮음(작은 청크가 여러 페이지 커버에 불리한 고전적 트레이드오프).
   순이득은 여전히 크게 양수.
6. **공통 실패 Q6·33·37·39**는 cs300 전 조합에서 불사신 — 정답 자체 추출/구조 문제로 추정,
   전처리 파트 진단 필요.

## 6. 결론 (§23: 2안 제시, 갱신)

| | 조합 | Hit@3 / MRR | 근거 |
|---|---|---|---|
| **최고 품질 (= 최종 채택안)** | **cs300 + rerank (rewrite 제외)** | **0.917 / 0.794** | 전 조합 중 최고, 가장 단순, 가장 빠름(4.2s) |
| **비용 대비 최적** | **cs300 + dedup만** | 0.883 / 0.680 | rerank도 없이 즉답, cs500 구 최고 조합(0.850) 상회 |

이번 라운드는 "최고 품질 = 비용도 최선"인 드문 경우 — rewrite 제거로 API 의존성과
응답시간이 함께 줄면서 성능도 올랐다. cs500 시절 결과(§3 위쪽 4행)는 비교 참고용으로 유지.
최종 확정은 LLM 풀 평가와 청킹 파트(chunk_size 재채택 여부) 협의로.

## 7. 다음 단계 / 팀 영향

- **팀장님:** cs300 기준 2안(dedup만 / rerank 단독) LLM 풀 평가 요청 — chunk_size
  변경과 rewrite 제거가 답변 지표(숫자 정확도 등)에도 반영되는지, rerank 4.2s 포함 총 응답시간
- **수민님:** ⚠️ **chunk_size 300 재채택 제안** — 기존 cs500 채택(baseline_report.md)은
  구 임베딩·구 평가셋 기준이었음, 현재 조건에서 cs300이 전반적으로 우세.
  조건나열형은 cs300+rerank로 부분 해결(0.64→0.82) — parent-child 실험(PR #41)과
  비교/병행 검토 요청. hybrid 로더는 cs300 결과로 폐기 권고
- **query rewrite(§ rewrite_rrf_effect.md) 폐기 제안** — cs300+rerank 조합에서 기여 0
  확인(§5-2). rag_chain.py의 rewrite_query/rrf_merge 함수는 코드에 남기되(다른 조합에서
  재검토 가능성 대비) evaluate.py 등 실사용 경로에서는 제외 권고
- 공통 실패 Q6·33·37·39 원인 진단 필요 (전처리/추출 관점)
- 과적합 방지: 최종 확정 후 테스트셋(별도 보관분) 1회 검증 필요 (§3·§23)

## 8. 재현 방법

```bash
# cs500 (참고용, 이전 라운드)
python src/eval_retrieval.py --run-name f66_dedup          --chunk-size 500 --overlap 100 --top-k 3
python src/eval_retrieval.py --run-name f66_hyb_rrf        --chunk-size 500 --overlap 100 --top-k 3 --loader hybrid --rewrite
python src/eval_retrieval.py --run-name f66_rrf_rerank     --chunk-size 500 --overlap 100 --top-k 3 --rewrite --rerank
python src/eval_retrieval.py --run-name f66_hyb_rrf_rerank --chunk-size 500 --overlap 100 --top-k 3 --loader hybrid --rewrite --rerank

# cs300
python src/eval_retrieval.py --run-name f66_cs300_dedup          --chunk-size 300 --overlap 100 --top-k 3
python src/eval_retrieval.py --run-name f66_cs300_hyb_rrf        --chunk-size 300 --overlap 100 --top-k 3 --loader hybrid --rewrite
python src/eval_retrieval.py --run-name f66_cs300_rerank_only    --chunk-size 300 --overlap 100 --top-k 3 --rerank
python src/eval_retrieval.py --run-name f66_cs300_rrf_rerank     --chunk-size 300 --overlap 100 --top-k 3 --rewrite --rerank
python src/eval_retrieval.py --run-name f66_cs300_hyb_rrf_rerank --chunk-size 300 --overlap 100 --top-k 3 --loader hybrid --rewrite --rerank

# 최종 채택안 (재현 시 이 한 줄로 충분)
python src/eval_retrieval.py --run-name repro_final --chunk-size 300 --overlap 100 --top-k 3 --rerank
```
