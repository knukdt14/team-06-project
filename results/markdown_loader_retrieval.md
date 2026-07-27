# markdown 로더 × 검색 레시피 조합 검증 — Q18↔Q6 맞바꾸기, 현행 유지 결론

- **작성:** 이희영 · 2026-07-28
- **목적:** 수민님 markdown 로더(PR #37, 표 보존)가 검색 파트의 유일한 미해결 문항
  Q18(장기주택저당차입금, 표 안에 정답)을 잡는지 — 기존 레시피와 2×2 조합 비교
- **관련:** [rewrite_rrf_effect.md](rewrite_rrf_effect.md)(직전 실험) · PR #37(markdown 로더)
  · 원본: `eval/retrieval_es26_md.csv`, `eval/retrieval_es26_rrf_md.csv`

---

## 1. 한눈에 (TL;DR)

- ✅ **Q18 해결 확인** — markdown 로더에서 정답 p.133이 2위로 검색됨 (표 보존 효과 입증)
- ⚠️ 그러나 **맞바꾸기 발생**: markdown은 Q6(신용카드)을 새로 실패시키고 **전반 MRR 하락**
  (0.833 → 0.717) — 표의 마크다운 변환으로 청크 구성이 바뀌며 다른 문항 순위가 밀림
- **결론: 현행 유지 (pypdf + rewrite+RRF)** — Hit@3 동률(0.957)에서 MRR 우세가 결정 근거
- 후속 카드: 두 로더가 **서로 다른 문항을 잘 잡음** → PyPDF+Markdown 이중 스토어 RRF 앙상블
  (수민님 handoff 제안 ② 원형)이 유망하나, 과적합 위험으로 팀 협의 후 진행
- 부수: PR #37의 `get_vectorstore_path` loader 인자 누락 버그 핫픽스 포함

## 2. 실험 조건

고정: bge-m3 · cs500/ov100 · top_k 3 · 26문항(답변가능 23) · 복수 정답 기준.
변경 변수 2개: 로더(pypdf/markdown) × 검색(dedup만 / rewrite+RRF).

## 3. 결과 (2×2)

| 조합 | Hit@3 | MRR | 실패 문항 |
|---|---|---|---|
| pypdf + dedup | 0.870 | 0.826 | Q14, Q18, Q19 |
| **pypdf + rewrite+RRF (현행)** | **0.957** | **0.833** | Q18 |
| markdown + dedup | 0.870 | 0.638 | Q6, Q10, Q19 |
| markdown + rewrite+RRF | 0.957 | 0.717 | Q6 |

### Q18 상세 (목표 문항)

| 로더 | 검색 결과 (gold [5, 133, 233]) |
|---|---|
| pypdf | 실패 — 표가 납작하게 추출돼 한도 숫자와 항목이 분리 |
| markdown | `[132, 133, 234]` — **p.133이 2위 hit** (표 구조 보존 덕) |

## 4. 해석

1. **로더는 문항별 강약을 맞바꾼다.** markdown은 표 기반 문항(Q18, Q14)을 살리지만
   Q6을 죽이고 전반 순위(MRR)를 떨어뜨림 — 표→마크다운 변환이 청크 경계를 바꿔
   기존에 잘 되던 문항들의 정답 청크 구성이 달라진 것.
2. **Hit@3 동률이면 MRR이 판정 기준** (§8 임베딩 비교 때와 동일한 논리) —
   정답 순위가 높을수록 LLM 문맥 품질이 좋아 답변 지표에도 유리할 가능성.
3. **rewrite+RRF는 로더와 무관하게 +0.087** (0.870→0.957 양쪽 동일) —
   질문 쪽 보강과 문서 쪽 보강이 독립적으로 작동함을 확인.

## 5. 한계

- 답변 지표 미확인 — markdown의 MRR 하락이 실제 숫자 정확도 하락으로 이어지는지는
  LLM 평가 필요 (다만 검색 지표 열세라 우선순위 낮음)
- Q6 실패 원인(마크다운 변환 후 신용카드 표의 청크 상태) 미진단 — 수민님 파트 공유
- 26문항 기준 — 표 기반 문항이 늘어나면 markdown 쪽 재평가 여지

## 6. 다음 단계 / 팀 영향

- **최종 검색 레시피 유지: bge-m3 + pypdf + rewrite+RRF + 페이지 dedup + top_k 3** (Hit@3 0.957)
- **수민님:** Q18은 markdown 로더로 해결 가능함이 입증됨(로더 자체는 유효한 도구).
  Q6의 마크다운 변환 후 상태 진단 요청. PR #37 loader 인자 버그는 본 PR에서 핫픽스
- **후속 검토(팀 협의):** PyPDF+Markdown 이중 스토어 RRF — 두 로더의 실패 문항이
  겹치지 않아(Q18 vs Q6) 이론상 전 문항 커버 가능. 단 조합이 늘수록 과적합 위험 →
  테스트셋 분리 후 검증하는 조건으로 제안
- **팀장님:** 현행 레시피 기준 LLM 평가 계속 진행하면 됨 (이번 결과로 변경 없음)

## 7. 재현 방법

```bash
# markdown 로더는 pymupdf4llm 필요: pip install -r requirements.txt
python src/eval_retrieval.py --run-name es26_md     --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --top-k 3 --loader markdown
python src/eval_retrieval.py --run-name es26_rrf_md --chunk-size 500 --overlap 100 --embedding huggingface --embedding-model BAAI/bge-m3 --top-k 3 --loader markdown --rewrite
# 비교 기준: retrieval_es26_dedup.csv, retrieval_es26_rrf.csv (rewrite_rrf_effect.md)
```
