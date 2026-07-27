# 페이지 dedup 효과 — bge-m3 전 문항 검색 성공 (13/13)

- **작성:** 이희영 · 2026-07-28
- **목적:** 같은 페이지 청크가 top-k를 독점하는 문제(Q8 `[58,58,58]`, Q11 `[181,181,181]`)를
  페이지 단위 중복 제거로 해결 — [handoff_ensemble.md](handoff_ensemble.md) 제안 ① 대응
- **관련:** [embedding_comparison.md](embedding_comparison.md)(직전 실험) · 원본: `eval/retrieval_emb_bgem3_dedup.csv`, `eval/retrieval_emb_kosroberta_dedup.csv`

---

## 1. 한눈에 (TL;DR)

- ⭐ **bge-m3 + dedup: Hit@3 0.923 → 1.000 (13/13 전 문항), MRR 0.923 → 0.962**
- 목표였던 Q8 해결: `[58,58,58]` → `[58, 10, 115]` (정답 p.10이 2위 진입)
- 부작용 없음 — 기존 hit 문항 전부 유지, top-3 내 중복 문장 수 감소 (Q2·Q11: 5→0)
- ⚠️ 13문항 만점은 **과적합 가능성** — 문항 확장(§3) 후 재검증 필요 (handoff_ensemble.md 경고와 동일)

## 2. 방법

- `rag_chain.py`에 `dedup_docs_by_page()` 추가: 검색 순위대로 훑으며 **처음 보는 페이지만** 채택
- `eval_retrieval.py`에 `--dedup` / `--fetch-k`(기본 15) 인자 추가:
  1차로 fetch_k개 검색 → 서로 다른 페이지 top_k개만 남겨 채점
- 고정 조건: cs500/ov100 · 머리말 제거 · 복수 정답 기준 · similarity · top_k 3 (직전 실험과 동일)

## 3. 결과

| 설정 | Hit@3 | MRR | 실패 문항 |
|---|---|---|---|
| bge-m3 (dedup 전) | 0.923 | 0.923 | Q8 |
| **bge-m3 + dedup** | **1.000** | **0.962** | **없음** |
| ko-sroberta (dedup 전) | 0.923 | 0.808 | Q10 |
| ko-sroberta + dedup | 0.923 | 0.808 | Q10 |

### 문항별 주요 변화 (bge-m3)

| 문항 | dedup 전 | dedup 후 | 비고 |
|---|---|---|---|
| Q8 (부양가족 기본공제) | `[58, 58, 58]` ❌ | `[58, 10, 115]` ✅ | 정답 p.10이 2위 진입 |
| Q11 (혼인세액공제) | `[181, 181, 181]` | `[181, 6, 39]` | 정답 페이지 2개(181·6) 확보 |
| Q2 (월세 요건) | 중복 문장 5 | 중복 문장 0 | LLM 전달 문맥 다양화 |

## 4. 해석

1. **dedup은 "중복형 실패"를 고친다** — Q8·Q11처럼 정답이 4위 밖으로 밀려난 경우,
   중복이 차지하던 슬롯을 비워 정답을 끌어올림.
2. **"미검색형 실패"에는 효과 없다 (대조군)** — ko-sroberta의 Q10은 top 15 안에도
   정답이 없는 유형이라 dedup으로 해결 불가, 점수 변화 없음. 두 실패 유형의 구분이
   명확해짐: 중복형 → dedup/MMR로, 미검색형 → 임베딩/청킹으로 해결.
3. **부수 효과** — top-3 내 중복 문장 감소로 같은 top_k에서 LLM이 받는 정보량이 늘어남.
   답변 지표(숫자 정확도·조건 포함률) 개선으로 이어지는지는 LLM 평가에서 확인 필요.

## 5. 한계

- **평가 문항 13개에서의 만점은 과적합일 수 있음** — 개발셋 확장(60~100문항, 팀장님 §3) 후 재검증 전까지 잠정 결과로 취급
- fetch_k=15 단일값만 실험 — top_k 실험(§9)과 함께 fetch_k 민감도도 확인 예정
- 페이지 기준이라 "다른 페이지의 중복 내용"은 못 거름 — 필요 시 내용 유사도 기반 dedup을 2차 후보로 보유

## 6. 다음 단계 / 팀 영향

- **현재 최선 검색 조합: bge-m3 + similarity + 페이지 dedup(fetch_k 15)** — 이후 실험의 기준선
- **다음 실험(희영):** top_k(§9) → MMR/hybrid 비교(§10~11). MMR은 dedup과 같은 문제를 푸는
  내장 기능이므로 "dedup을 이기는지"가 §10의 핵심 비교 포인트
- **팀장님:** LLM 풀 평가 시 `--dedup` 조건 포함 요청 (evaluate.py에는 아직 dedup 미적용 —
  적용 원하시면 eval_retrieval.py와 같은 패턴으로 추가 가능, 협의 필요)
- **수민님:** 제안 ① 완료. 제안 ②(PyPDF+Markdown RRF 앙상블)는 후속 검토

## 7. 재현 방법

```bash
python src/eval_retrieval.py --run-name emb_bgem3_dedup --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --dedup
python src/eval_retrieval.py --run-name emb_kosroberta_dedup --chunk-size 500 --overlap 100 --dedup
```
