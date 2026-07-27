# 검색 방식 3파전 — similarity+dedup 최종 확정 (가이드라인 §10~11)

- **작성:** 이희영 · 2026-07-28
- **목적:** similarity+페이지 dedup(현 기준선) vs MMR vs hybrid 비교로 최종 검색 방식 확정
- **관련:** [dedup_effect.md](dedup_effect.md) · [topk_effect.md](topk_effect.md) ·
  원본: `eval/retrieval_mmr_l*.csv`, `eval/retrieval_hyb_b*.csv`

---

## 1. 한눈에 (TL;DR)

- ⭐ **최종 검색 레시피 확정: bge-m3 + similarity + 페이지 dedup + top_k 2~3** (도전자 전패)
- MMR(λ 0.2/0.5/0.8 전부): Hit@3 0.923 — Q8을 못 잡음 (내용 기반 다양화의 한계)
- hybrid: BM25 비중이 높을수록 MRR 급락(0.923→0.346) — **일상어 질문에 키워드 검색이 역효과**,
  프로젝트 주제 선정 이유("의미 검색이 필요한 도메인")를 실험으로 입증
- 코드 작업: MMR `fetch_k`/`lambda_mult`, hybrid `HYBRID_WEIGHTS`·`--bm25-weight` 인자 추가 (§10 요구사항)

## 2. 실험 조건

| 항목 | 값 |
|---|---|
| 고정 | bge-m3 · cs500/ov100 · 머리말 제거 · 복수 정답 기준 · top_k 3 |
| MMR | fetch_k 20, lambda_mult 0.2 / 0.5 / 0.8 |
| hybrid | BM25:Dense = 0.2:0.8 / 0.5:0.5 / 0.8:0.2 (EnsembleRetriever) |
| 기준선 | similarity + 페이지 dedup(fetch_k 15) — [dedup_effect.md](dedup_effect.md) |

## 3. 결과

| 방식 | Hit@3 | MRR | 실패 문항 |
|---|---|---|---|
| **similarity + dedup (기준선)** | **1.000** | **0.962** | 없음 |
| MMR λ0.2 | 0.923 | 0.923 | Q8 |
| MMR λ0.5 | 0.923 | 0.923 | Q8 |
| MMR λ0.8 | 0.923 | 0.923 | Q8 |
| hybrid BM25 0.2 | 0.923 | 0.923 | Q8 |
| hybrid BM25 0.5 | 0.923 | 0.538 | Q8 |
| hybrid BM25 0.8 | 0.923 | 0.346 | Q8 |

※ MMR 세 λ의 총점이 같지만 문항별 검색 페이지는 12/13개가 상이 — λ 전달 정상 확인.
※ hybrid는 EnsembleRetriever가 두 검색의 합집합을 반환해 top 3보다 많은 문서로 채점됨
(채점이 다소 유리한 조건인데도 기준선에 미달).

## 4. 방식별 패배 원인

1. **MMR — 내용 기반 다양화의 한계.** Q8에서 `[58, 115, 236]`처럼 다양화는 되지만
   정답 페이지(4·10·113·117)를 top 3에 못 올림. MMR은 "내용이 비슷한" 청크만 걸러내므로
   같은 페이지의 서로 다른 항목 청크는 통과됨 → 페이지 번호로 자르는 dedup이 이 문서에선 더 확실.
   λ0.8에서 Q11이 `[181,181,181]`로 회귀 — λ↑일수록 similarity와 동일해지는 패턴 확인.
2. **hybrid — 일상어 질문에 BM25 역효과.** "얼마나 돌려받아요?" 같은 일상어는 문서 용어와
   단어가 겹치지 않아 BM25가 무관 페이지를 상위로 올림 → BM25 비중과 MRR이 정확히 반비례.
   키워드·숫자가 그대로 담긴 질문(예: 고유 용어 검색)이 늘어나면 재평가 여지는 있음.
3. **공통 실패 Q8의 정체** — 세 방식 모두 p.58을 1위로 올림: 임베딩 자체가 질문과 p.58을
   혼동하는 문제로, 검색 방식 변경으로는 해결 불가. reranker(§12)의 후보 사례.

## 5. 한계

- 13문항 기준 — 문항 확장(§3) 후 재검증 필요. 특히 hybrid는 키워드형 질문 추가 시 재평가 여지
- hybrid의 RRF(순위 융합) 방식은 미실험 — handoff_ensemble.md 제안 ②와 묶어 후속 검토 가능
- 검색 지표 기준 결론 — 최종 확인은 LLM 평가(숫자 정확도)로

## 6. 다음 단계 / 팀 영향

- **§8~11 완료 — 검색 파트 확정안: bge-m3 + similarity + 페이지 dedup(fetch_k 15) + k 2~3**
- **팀장님:** 이 확정 조합으로 LLM 풀 평가 진행 요청 (k=2 vs 3 비교 포함).
  이후 프롬프트·LLM 비교(§14~15)는 이 검색 설정 고정을 전제로 하면 됨
- **config 기본값 갱신 협의:** HF_EMBEDDING_MODEL → bge-m3, SEARCH_TYPE 유지(similarity),
  dedup 기본 적용 여부는 rag_chain(main.py 챗봇)까지 반영할지 팀 결정 필요
- (여유 시) reranker로 Q8 유형 해결(§12), RRF 앙상블(수민님 제안 ②), e5 prefix 검증

## 7. 재현 방법

```bash
# MMR (fetch_k 20 고정, λ 변경)
python src/eval_retrieval.py --run-name mmr_l02 --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --search-type mmr --top-k 3 --fetch-k 20 --mmr-lambda 0.2
python src/eval_retrieval.py --run-name mmr_l05 --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --search-type mmr --top-k 3 --fetch-k 20 --mmr-lambda 0.5
python src/eval_retrieval.py --run-name mmr_l08 --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --search-type mmr --top-k 3 --fetch-k 20 --mmr-lambda 0.8

# hybrid (BM25 비중 변경, Dense = 1-값)
python src/eval_retrieval.py --run-name hyb_b02 --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --search-type hybrid --top-k 3 --bm25-weight 0.2
python src/eval_retrieval.py --run-name hyb_b05 --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --search-type hybrid --top-k 3 --bm25-weight 0.5
python src/eval_retrieval.py --run-name hyb_b08 --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --search-type hybrid --top-k 3 --bm25-weight 0.8
```
