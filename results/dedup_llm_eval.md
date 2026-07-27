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

## 3. 원인 규명 1차: temperature=0 누락 발견

문항별로 pre/post를 대조한 결과, Upstage의 numeric_score가 떨어진 두 문항의 원인이 서로 달랐다.

| 문항 | 검색된 페이지(dedup 전) | 검색된 페이지(dedup 후) | 원인 |
|---|---|---|---|
| Q6 | `[150,156,154]` | `[150,156,154]` **(동일!)** | dedup과 무관 — 순수 LLM 샘플링 변동 |
| Q13 | `[168,169,168]` | `[168,169,177]` | dedup이 중복 슬롯을 다른 페이지로 교체 → 진짜 효과 |

Q6은 **완전히 같은 문맥을 주고도 답이 달라진 사례**다. 원인을 코드에서 찾았다:
`rag_chain.py`의 `get_llm()`이 Gemini·OpenAI엔 `temperature=0`을 주면서 **Upstage에는
빠뜨려져 있었다.** 즉 Upstage는 지금까지의 모든 실험에서 답변이 매번 달라질 수 있는
상태였다. **수정 완료** (`ChatUpstage(..., temperature=0)`).

## 3-1. 수정 후 재실행 (10:13) — temperature=0으로도 완전한 결정성은 아니었다

| | dedup 전 | dedup 후 (버그 상태, 09:56) | **dedup 후 (수정 후, 10:13)** |
|---|---|---|---|
| numeric_acc | 0.9231 | 0.8462 | **0.8718** |
| condition_recall | 0.9231 | 0.9231 | 0.8846 |
| Q8 (dedup 목표 문항) | 실패("찾을 수 없음") | - | **✅ 정답** (150만원·100만원 모두 포함, 검색 `[58,10,115]`) |

**목표였던 Q8은 완전히 해결됐다** — dedup의 존재 이유가 실증됨.

수정 후에도 값이 조금 바뀌었는데(0.8462→0.8718), 재확인해보니 Q4·Q6·Q13 **세 문항 모두
검색된 페이지가 이전 실행과 완전히 동일**했다(dedup 전/후 비교였던 Q13도 이번엔 페이지가
같았음). 즉 **temperature=0을 줘도 상용 LLM API가 100% 결정론적이진 않다** — 이는
재현성 실험(repro_cs500.md)에서 Gemini도 동일 설정 반복 시 문항 1~2개가 흔들렸던 것과
같은 종류의 잔여 노이즈다. Gemini 기준 확보된 노이즈 폭(numeric_acc ±0.026)과 비교하면
이번 변동(±0.026~0.05)은 그 경계 근처 — dedup의 순수 효과와 완전히 분리하려면 반복
실행이 필요하지만, **검색 완결성 확보(Q8 해결)라는 핵심 목표는 노이즈와 무관하게
확실하다.**

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

- **현재 최선 조합: bge-m3 + dedup(fetch_k=15) + Upstage** — 검색 13/13(Q8 해결 확인),
  numeric_acc 0.872, condition_recall 0.885, 가장 빠름(2.30s). temperature 버그 수정 반영된
  수치로 확정.
- OpenAI(gpt-5.4-mini)는 dedup과 궁합이 가장 좋음(numeric_acc 0.872, Upstage와 동률) — 유력한 대안.
- Gemini는 종합 지표상 우위가 없고 응답시간도 가장 느림(6.58s) — 이번 비교에서는 후순위.
- Upstage·OpenAI 둘 다 numeric_acc 0.872로 동률 → 응답시간·조건포함률 등 다른 기준으로 추가 비교 필요.

## 6. 다음 할 일

- [x] ~~Upstage temperature 수정 반영 후 llm_bgem3_upstage_dedup 재실행~~ 완료 (10:13, numeric_acc 0.872)
- [x] ~~fetch_k 민감도 확인~~ → **이희영이 top_k 실험으로 상위 호환 완료** (results/topk_effect.md):
      k=2에서 이미 검색 포화(Hit@k 1.000), k=8·10은 이득 없이 노이즈만 증가 → fetch_k 자체보다
      top_k(k=2 vs k=3)가 실질 변수임이 확인됨. 이 항목은 아래 신규 항목으로 대체.
- [ ] **⭐ k=2 vs k=3 LLM 풀 평가 (이희영 직접 요청)** — bge-m3+dedup 고정,
      본 문서의 최종 후보 Upstage·OpenAI 두 모델에 대해 top_k만 바꿔 비교:
      ```bash
      python src/evaluate.py --run-name k2_upstage --embedding-model BAAI/bge-m3 --llm upstage --dedup --top-k 2
      python src/evaluate.py --run-name k3_upstage --embedding-model BAAI/bge-m3 --llm upstage --dedup --top-k 3
      python src/evaluate.py --run-name k2_openai  --embedding-model BAAI/bge-m3 --llm openai  --dedup --top-k 2
      python src/evaluate.py --run-name k3_openai  --embedding-model BAAI/bge-m3 --llm openai  --dedup --top-k 3
      ```
      k=2는 입력 토큰이 k=3 대비 약 2/3라 응답시간·비용까지 함께 확보 가능 — 검색 지표로는
      k=2·3 무승부이므로 **숫자 정확도·조건 포함률 차이로 최종 k를 결정**(§9 원칙)
- [ ] 위 결과로 **Upstage vs OpenAI 동률(0.872) 타이브레이커까지 한 번에 해결** —
      k별 numeric_acc 우세 모델 + 응답시간을 종합해 최종 LLM·k 확정
- [ ] temperature=0에서도 남는 잔여 노이즈 확인 — 최종 조합 확정 후 2~3회 반복해 노이즈 폭 확정
- [ ] Gemini Q10 개별 diagnosis (문맥은 정답인데 왜 거절하는지) — 우선순위 낮음(Gemini는 이미 후순위)
- [ ] (희영 다음 실험) similarity+dedup vs MMR vs hybrid 비교(§10~11) — k=2·3 후보값으로 진행 예정

## 7. 재현 명령

```bash
python src/evaluate.py --run-name llm_bgem3_upstage_dedup --embedding-model BAAI/bge-m3 --llm upstage --dedup
python src/evaluate.py --run-name llm_bgem3_gemini_dedup  --embedding-model BAAI/bge-m3 --llm gemini  --dedup
python src/evaluate.py --run-name llm_bgem3_openai_dedup  --embedding-model BAAI/bge-m3 --llm openai  --dedup
```
