# dedup을 실제 답변 생성에 연결 + LLM별 효과 확인

- **작성:** 이수홍 · 2026-07-27
- **배경:** 이희영의 dedup_effect.md 6번 요청 — "LLM 풀 평가 시 --dedup 조건 포함,
  evaluate.py에는 아직 미적용"
- **관련:** [dedup_effect.md](dedup_effect.md) · [scoring_bugfix.md](scoring_bugfix.md)

---

## 1. 무엇을

`rag_chain.get_retriever`/`build_chain`에 dedup·fetch_k를 공용화해 `evaluate.py --dedup`으로
실제 LLM 답변 생성 파이프라인에서도 페이지 dedup을 쓸 수 있게 연결. bge-m3 기준으로
Gemini·OpenAI(gpt-5.4-mini)·Upstage 3개 LLM에 dedup 전/후 풀 평가.

## 2. 결과 — 검색은 전 모델 13/13 달성

| | 검색 Hit@3 | numeric_acc | condition_recall | wrong_refusals |
|---|---|---|---|---|
| Gemini (dedup 전) | 0.923 | 0.808 | 0.782 | 1 |
| **Gemini (dedup 후)** | **1.000** | 0.821 | 0.808 | 1 |
| OpenAI-mini (dedup 전) | 0.923 | 0.782 | 0.885 | 0 |
| **OpenAI-mini (dedup 후)** | **1.000** | **0.872** | 0.769 | 0 |
| Upstage (dedup 전) | 0.923 | **0.923** | **0.923** | 0 |
| **Upstage (dedup 후)** | **1.000** | 0.846 | 0.923 | 0 |

dedup_effect.md의 예측대로 **검색은 3개 LLM 전부 13/13(E3 소멸)**을 달성했다.
그러나 numeric_acc는 모델마다 반응이 달랐다 — OpenAI는 개선(+0.09), Gemini는 미미한
개선(+0.01), **Upstage는 오히려 하락(-0.08)**.

## 3. 원인 규명: Upstage 하락은 dedup 때문이 아니었다

문항별로 pre/post를 대조한 결과, Upstage의 numeric_score가 떨어진 두 문항의 원인이 서로 달랐다.

| 문항 | 검색된 페이지(dedup 전) | 검색된 페이지(dedup 후) | 원인 |
|---|---|---|---|
| Q6 | `[150,156,154]` | `[150,156,154]` **(동일!)** | dedup과 무관 — 순수 LLM 샘플링 변동 |
| Q13 | `[168,169,168]` | `[168,169,177]` | dedup이 중복 슬롯을 다른 페이지로 교체 → 진짜 효과 |

Q6은 **완전히 같은 문맥을 주고도 답이 달라진 사례**다. 원인을 코드에서 찾았다:
`rag_chain.py`의 `get_llm()`이 Gemini·OpenAI엔 `temperature=0`을 주면서 **Upstage에는
빠뜨려져 있었다.** 즉 Upstage는 지금까지의 모든 실험에서 답변이 매번 달라질 수 있는
상태였다. **수정 완료** (`ChatUpstage(..., temperature=0)`).

→ Upstage의 실제 dedup 순효과는 Q6을 제외하면 **거의 중립**(Q13만 -1항목).
검색 완결성(13/13)은 확보했고 답변 품질 손실은 이전에 알려진 것보다 훨씬 작다.

## 4. 이 결과의 의미

1. **temperature 버그가 지금까지의 모든 Upstage 실험(llm_upstage, llm_bgem3_upstage 등)에
   재현성 오차를 심었을 수 있다.** 재현성 실험(repro_cs500.md)은 Gemini로만 했으므로
   Upstage의 노이즈 폭은 아직 검증되지 않음 — 수정 후 재실행 권장.
2. **dedup은 여전히 순검색 지표 개선 효과가 확실하다** (E3 완전 소멸). 답변 품질 손실은
   미미하거나(Upstage) 오히려 이득(OpenAI)이라, dedup 채택을 뒤집을 근거는 아니다.
3. **Gemini의 Q10 거절(WR:1)은 dedup 후에도 그대로.** 검색이 완벽해져도(13/13) 여전히
   거절한다는 뜻 — 순수 생성측 문제로 재확인. Q10만 놓고 Gemini 프롬프트를 따로
   점검할 가치가 있다.
4. **fetch_k=15는 미세 검증 대상으로 남음.** Q13처럼 "이미 맞는 중복 슬롯"을 다른
   페이지로 갈아 끼우는 부작용이 있어, 더 작은 fetch_k(6~10)로도 Q8이 해결되는지
   `eval_retrieval.py`(API 불필요)로 먼저 확인해볼 가치가 있다.

## 5. 잠정 결론

- **현재 최선 조합: bge-m3 + dedup(fetch_k=15) + Upstage** — 검색 13/13, numeric_acc 0.846,
  가장 빠름(2.32s). temperature 수정 후 재실행해 수치를 확정할 것.
- OpenAI(gpt-5.4-mini)는 dedup과 궁합이 가장 좋음(numeric_acc 0.872, 최고) — 유력한 대안.
- Gemini는 종합 지표상 우위가 없고 응답시간도 가장 느림(6.58s) — 이번 비교에서는 후순위.

## 6. 다음 할 일

- [ ] Upstage temperature 수정 반영 후 llm_bgem3_upstage_dedup 재실행 (진짜 수치 확정)
- [ ] fetch_k 민감도 확인 (`eval_retrieval.py --dedup --fetch-k 6/8/10`, API 불필요)
- [ ] Gemini Q10 개별 diagnosis (문맥은 정답인데 왜 거절하는지)

## 7. 재현 명령

```bash
python src/evaluate.py --run-name llm_bgem3_upstage_dedup --embedding-model BAAI/bge-m3 --llm upstage --dedup
python src/evaluate.py --run-name llm_bgem3_gemini_dedup  --embedding-model BAAI/bge-m3 --llm gemini  --dedup
python src/evaluate.py --run-name llm_bgem3_openai_dedup  --embedding-model BAAI/bge-m3 --llm openai  --dedup
```
