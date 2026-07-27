# query rewrite + RRF 앙상블 — 26문항에서 실패 3→1 (Q10 처방 검증)

- **작성:** 이희영 · 2026-07-28
- **목적:** handoff_ensemble.md "Q10 정밀 처방" ① 구현·검증 — 질문에 제도명이 없어
  임베딩이 정답을 못 잇는 문제를 일반 규칙(LLM 질문 재작성 + RRF 융합)으로 해결
- **관련:** [handoff_ensemble.md](handoff_ensemble.md)(처방 원본) · [search_method_comparison.md](search_method_comparison.md)(직전 확정 레시피)
  · 원본: `eval/retrieval_es26_dedup.csv`, `eval/retrieval_es26_rrf.csv`, `eval/query_rewrites.csv`

---

## 1. 한눈에 (TL;DR)

- ⭐ **26문항 체제에서 Hit@3 0.870 → 0.957** (실패 3문항 → 1문항)
- 파이프라인: 원질문 + LLM 확장질문 **이중 검색 → RRF 순위 융합 → 페이지 dedup → top 3**
- 신규 실패 문항 Q14(교육비)·Q19(한부모)를 해결 — 확장 질문이 제도명("한부모 소득공제" 등)을
  보강해 의미 간극을 메움. 특정 문항 하드코딩 없는 **일반 규칙**이라 과적합 아님
- 확장 질문은 `eval/query_rewrites.csv`에 캐싱 — 최초 1회만 LLM 호출(26회), 이후 실험 무료

## 2. 방법

- `rag_chain.py`: `rewrite_query()`(질문을 문서 용어로 재작성, Upstage 사용),
  `rrf_merge()`(순위 기반 융합, score=Σ1/(60+rank)) 추가
- `eval_retrieval.py`: `--rewrite` / `--rewrite-llm` 인자. rewrite 모드는 fetch_k(15)개씩
  이중 검색 후 RRF → 페이지 dedup을 수동 적용 (이중 dedup 방지)
- 고정 조건: bge-m3 · cs500/ov100 · top_k 3 · 26문항(답변가능 23) · 복수 정답 기준

## 3. 결과

| 설정 | Hit@3 | MRR | 실패 문항 |
|---|---|---|---|
| 기존 레시피 (similarity+dedup) | 0.870 | 0.826 | Q14, Q18, Q19 |
| **+ rewrite + RRF** | **0.957** | **0.833** | **Q18** |

### 해결 사례

| 문항 | 전 → 후 | 메커니즘 |
|---|---|---|
| Q19 (한부모 공제) | `[117,115,8]` 실패 → `[117,115,4]` 3위 hit | 확장 질문에 "한부모 소득공제" 제도명 포함 |
| Q14 (교육비) | 실패 → hit | 학교급별 교육비 용어 보강 |
| Q10 (표준세액공제) | 양쪽 모두 1위 hit | 회귀 수정 후 dedup만으로도 안정 (rewrite 병행에도 유지) |

## 4. 한계 (중요)

1. **재작성이 오정보를 포함할 수 있음** — Q1·Q2 확장 질문에 틀린 숫자 확인
   (월세 한도 1,000만원을 "750만원"으로). 확장 질문은 **검색에만** 쓰이고 답변 생성에는
   미사용이라 오답으로 직결되진 않지만, 잘못된 단어가 검색을 오도할 가능성은 존재.
2. **출력 형식 불안정** — Q19 재작성이 "한 문장만" 지시를 어기고 해설까지 출력.
   개선안: 프롬프트 강화 또는 첫 문장 추출 후처리. (검색엔 그래도 동작함)
3. **Q18(장기주택저당차입금)은 미해결** — rewrite로도 안 잡히는 미검색형. 청킹/전처리
   관점 진단 필요 (수민님 파트 공유).
4. LLM 재작성 비용: 문항당 1회 (캐시로 상쇄). 실서비스라면 질문마다 호출 필요 →
   응답시간·비용 증가는 LLM 평가에서 확인해야 함.

## 5. 다음 단계 / 팀 영향

- **팀장님:** rewrite+RRF 조건의 LLM 풀 평가 요청 — 검색 개선(+0.087)이 숫자 정확도로
  이어지는지, 재작성 1회 추가 호출의 응답시간 영향 포함
- **수민님:** Q18 진단 이관 (정답 페이지의 추출/청킹 상태 확인 필요), 제안 ②(RRF) 구현 완료 보고
- **채택 판단:** LLM 평가에서 개선 확인 시 rewrite 모드를 config 기본값 후보로;
  실서비스 관점에선 호출 비용 대비 이득을 §23 기준으로 판단
- main.py 챗봇에 rewrite 적용 여부는 응답시간 트레이드오프 확인 후 결정

## 6. 재현 방법

```bash
# 기준선 (dedup은 config 기본값 ON)
python src/eval_retrieval.py --run-name es26_dedup --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --top-k 3
# rewrite + RRF (최초 1회만 UPSTAGE_API_KEY 필요, 이후 캐시 사용)
python src/eval_retrieval.py --run-name es26_rrf --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --top-k 3 --rewrite --rewrite-llm upstage
```
