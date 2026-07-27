# top_k 실험 — k=2에서 검색 포화, LLM 평가 후보 k=2·3 (가이드라인 §9)

- **작성:** 이희영 · 2026-07-28
- **목적:** 확정 검색 조합(bge-m3 + 페이지 dedup) 고정 상태에서 top_k만 변경,
  검색 지표 기준 최적 k 후보를 압축해 LLM 평가로 전달
- **관련:** [dedup_effect.md](dedup_effect.md)(직전 실험·기준선) · 원본: `eval/retrieval_k1.csv` ~ `retrieval_k10.csv`

---

## 1. 한눈에 (TL;DR)

- ⭐ **k=2에서 이미 전 문항 검색 성공(Hit@2 1.000)** — 이후 k를 늘려도 검색 지표 변화 없음
- k=1만 Q8 실패 (정답이 2위라 1개만 가져오면 놓침)
- **LLM 평가 후보: k=2 (최소·효율), k=3 (안전 마진)** — 최종 선택은 숫자 정확도 평가로
- k=8·10은 검색 이득 0 + 무관 청크 6~8개가 LLM에 전달됨 → 제외 권고

## 2. 실험 조건 (고정값)

| 항목 | 값 |
|---|---|
| 임베딩 | BAAI/bge-m3 (§8 1순위) |
| 검색 | similarity + 페이지 dedup (fetch_k 15, k8·10은 30) |
| chunk_size / overlap | 500 / 100 · 머리말 제거 · 복수 정답 기준 |
| 변경 변수 | top_k = 1 / 2 / 3 / 5 / 8 / 10 |

## 3. 결과

| top_k | Hit@k | MRR | 실패 문항 |
|---|---|---|---|
| 1 | 0.923 | 0.923 | Q8 |
| **2** | **1.000** | **0.962** | 없음 |
| 3 | 1.000 | 0.962 | 없음 |
| 5 | 1.000 | 0.962 | 없음 |
| 8 | 1.000 | 0.962 | 없음 |
| 10 | 1.000 | 0.962 | 없음 |

※ MRR이 k와 무관하게 일정한 이유: MRR은 "정답이 처음 등장하는 순위"만 반영 —
bge-m3+dedup은 13문항 중 12개에서 정답을 1위, Q8만 2위에 올리므로 k≥2면 값이 고정됨.

## 4. 해석

1. **검색은 k=2에서 포화** — 정답이 거의 전부 1~2위에 오르는 조합이라 k를 늘릴 검색상 이유가 없음.
2. **k 증가의 비용** (가이드라인 §9): 무관 청크 혼입 → 서로 다른 숫자가 섞여 LLM 혼동 위험,
   입력 토큰·응답 시간 증가. k=10이면 청크 10개 중 8개가 노이즈.
3. **k=3의 가치**: 평가셋에 "여러 페이지가 필요한 질문"(§3 유형)이 추가되면
   여유 슬롯이 Recall에 기여할 수 있음 → k=2와 함께 후보로 유지.

## 5. 한계

- 13문항 기준 — 문항 확장(60~100개) 후 포화 지점이 뒤로 밀릴 수 있음 (특히 멀티페이지 질문 추가 시)
- 검색 지표만의 결론 — **최종 k 확정은 LLM 평가(숫자 정확도·faithfulness)로** (§9 "Recall 최고 k가 아니라 최종 정확도 좋은 k")

## 6. 다음 단계 / 팀 영향

- **팀장님:** LLM 풀 평가 시 k=2 vs k=3 비교 요청 (bge-m3 + dedup 고정) —
  입력 토큰이 k=3 대비 약 2/3라 응답시간·비용 데이터도 함께 확보 가능
- **다음 실험(희영):** similarity+dedup vs MMR vs hybrid 비교(§10~11) —
  MMR·hybrid 실험 시 top_k는 후보값(2·3)으로 진행
- 평가 문항 확장(§3) 후 본 실험 재검증 필요

## 7. 재현 방법

```bash
# k1~k5는 fetch_k 기본값(15), k8·k10은 dedup 후 페이지 수 확보를 위해 30
python src/eval_retrieval.py --run-name k1  --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --dedup --top-k 1
python src/eval_retrieval.py --run-name k2  --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --dedup --top-k 2
python src/eval_retrieval.py --run-name k3  --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --dedup --top-k 3
python src/eval_retrieval.py --run-name k5  --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --dedup --top-k 5
python src/eval_retrieval.py --run-name k8  --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --dedup --top-k 8 --fetch-k 30
python src/eval_retrieval.py --run-name k10 --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --dedup --top-k 10 --fetch-k 30
```
