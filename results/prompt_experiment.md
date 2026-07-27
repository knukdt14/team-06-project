# 프롬프트 실험 — basic vs simple vs reasoning (reasoning 최종 채택)

- **작성:** 이수홍 · 2026-07-27
- **목적:** 가이드라인 §14 프롬프트 실험 — 검색·LLM이 확정된 최종 파이프라인
  (bge-m3 + similarity+dedup + top_k=3 + Upstage) 위에서 프롬프트만 바꿔 비교
- **관련:** [final_model_selection.md](final_model_selection.md) (basic 기준선, 4회 반복 노이즈 밴드)

---

## 1. 재현성 확인 — simple/reasoning 각 3회 반복

| 실행 | prompt | numeric_acc | numeric_perfect | BERTScore | 응답시간 |
|---|---|---|---|---|---|
| 1 | simple | 0.8975 | 0.8462 | 0.6807 | 3.45s |
| 2 | simple | 0.8975 | 0.8462 | 0.6832 | 3.33s |
| 3 | simple | 0.8975 | 0.8462 | 0.6799 | 3.41s |
| 1 | reasoning | 0.8462 | 0.7692 | 0.7417 | 1.59s |
| 2 | reasoning | 0.8205 | 0.7692 | 0.7413 | 1.55s |
| 3 | reasoning | 0.8462 | 0.7692 | 0.7477 | 1.66s |
| **basic (참고, 기존 4회)** | basic | 0.853 (±0.026) | 0.7692(고정) | 0.712 (±0.016) | 2.10s |

- simple: numeric_acc·numeric_perfect·error_codes가 3회 **완전히 동일** — 순수 노이즈가 아니라
  구조적 효과. BERTScore(0.680~0.683)·응답시간(3.33~3.45s)도 basic 노이즈 밴드 밖으로
  일관되게 벗어남.
- reasoning: BERTScore(0.741~0.748)·응답시간(1.55~1.66s)이 basic 밴드 밖으로 일관되게
  벗어남(개선). numeric_acc(0.82~0.85)는 basic과 겹치는 범위 — Upstage API 잔여 노이즈
  (final_model_selection.md에서 이미 확인된 성격)로 판단.

## 2. 문항 단위 비교 — 어디서 차이가 나는가

| id | basic | simple | reasoning |
|---|---|---|---|
| Q6 (300만원;250만원) | 0.333 (둘 다 누락) | **0.667 (300만원 회복)** | 0.333 (basic과 동일) |
| Q10 (13만원, 표준세액공제) | 0 (오답) | **0 (동일하게 오답)** | 0 (동일하게 오답) |
| Q13 (5년) | 0.667 (누락) | **1.000 (완전 정답)** | 0.667 (basic과 동일) |
| 나머지 10문항 | 전부 정답 | 전부 정답 | 전부 정답 |

- **Q10은 세 프롬프트에서 완전히 동일하게 오답** — 프롬프트 문제가 아니라 검색/문서 자체의
  문제로 확인됨. final_model_selection.md에 남겨둔 "Q10 개별 diagnosis"가 다음 진단 대상.
- simple의 numeric_acc 개선은 Q6·Q13 두 문항에서만 발생, reasoning은 이 두 문항에서
  basic과 완전히 같은 실수를 반복 — 즉 reasoning은 답의 실질 내용이 basic과 동일하고
  표현·속도만 개선된 것.

## 3. 트레이드오프 정리

| | 정답 내용 | 문체(BERTScore) | 응답시간 |
|---|---|---|---|
| simple | basic보다 실제 개선(Q6/Q13, 3회 재현) | 뚜렷이 나쁨 | 뚜렷이 느림(+65%) |
| reasoning | basic과 완전 동일 | 뚜렷이 좋음 | 뚜렷이 빠름(-25%) |

simple은 numeric_acc를 올리는 유일한 후보지만, basic/reasoning에 있던 "문맥에 없으면
문서에서 찾을 수 없다고 답하라" 지시가 빠져 있어 문서 밖 질문에서 할루시네이션 위험이
있고(이번 13문항엔 답변 불가 문항이 없어 직접 관측되진 않음), 응답시간도 유의미하게 늘어남.

## 4. 결론 — reasoning 채택

reasoning은 numeric_acc 기준으로 basic 대비 **손해가 전혀 없고**(같은 문항에서 같은 실수),
그 위에 BERTScore·응답시간을 동시에 개선한다. simple의 개선은 실재하지만
"거절 지시 부재" 리스크와 응답시간 증가라는 대가가 있어, 실사용 챗봇 관점에서 더
안전한 reasoning을 최종 채택한다. (이 판단은 수치가 아니라 가치 판단이 섞여 있으므로
발표 시 근거로 명시할 것 — dedup 채택 때와 같은 성격.)

`config.PROMPT_NAME`을 `"basic"` → `"reasoning"`으로 변경, `build_chain`/`evaluate.py`/
`main.py`/`rag_chain.py` CLI 기본값 모두 연동.

## 5. 남은 할 일

- [ ] Q10(13만원/표준세액공제) 개별 diagnosis — 프롬프트 무관 오답, 원문 p.187/410 재확인 필요
- [ ] 평가셋 확대 (13→40+, 과적합 리스크 여전)
- [ ] 발표 스토리 정리

## 6. 재현 명령

```bash
python src/evaluate.py --run-name prompt_reasoning_check --prompt reasoning
# (--prompt 생략 시 config.PROMPT_NAME 기본값으로 reasoning 실행됨)
```
