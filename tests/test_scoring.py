import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from scoring import (is_refusal, is_full_refusal, numeric_score, condition_score,
                     cited_pages, citation_score, tag_errors)


# ── 거절 판정 (관대 기준: 문구만 검사) ──
def test_refusal_detected():
    assert is_refusal("문서에서 찾을 수 없습니다.") is True

def test_refusal_normal_answer():
    assert is_refusal("월세액의 17%를 공제받습니다.") is False


# ── 전체 거절 판정 (엄격 기준: 문구 + 짧은 길이 → WR용) ──
def test_full_refusal_short_template():
    assert is_full_refusal("제공된 연말정산 안내 문서에서 확인할 수 없습니다.") is True

def test_full_refusal_false_for_long_partial_answer():
    # 실측 사례(gpt-5.4-nano Q6, 606자): 계산식·공제율을 상세히 답하면서
    # 한 문장만 "찾을 수 없다"고 헤지 — 전체 거절이 아니라 부분 답변이어야 한다
    long_partial = (
        "신용카드 등 소득공제 금액은 다음 산식에 따라 계산합니다. "
        "신용카드 등 소득공제 금액 = (① + ② + ③ + ④ + ⑤ - ⑥ + ⑦)에 해당하는 금액입니다. "
        "① 신용카드사용분: 신용카드 등 사용금액 합계액에서 대중교통이용분과 전통시장사용분을 제외한 금액입니다. "
        "다만 세부 항목 중 일부 세율은 본 문맥에서 확인할 수 없습니다. "
        "나머지 항목은 문맥에 제시된 산식을 그대로 적용하면 됩니다."
    )
    assert len(long_partial) > 120
    assert is_refusal(long_partial) is True       # 문구는 있음
    assert is_full_refusal(long_partial) is False  # 하지만 전체 거절은 아님

def test_full_refusal_false_for_normal_answer():
    assert is_full_refusal("월세액의 17%를 공제받습니다.") is False


# ── 핵심 숫자 (BERTScore가 못 잡는 오답을 잡아야 함) ──
def test_numeric_all_present():
    score, missing = numeric_score("연 13만원을 공제받습니다.", "13만원")
    assert score == 1.0 and missing == []

def test_numeric_wrong_amount():
    # Q10류 오답: 13만원을 7만원이라 답함 → 0점이어야 한다
    score, missing = numeric_score("연 7만원을 공제받습니다.", "13만원")
    assert score == 0.0 and missing == ["13만원"]

def test_numeric_comma_variant():
    score, _ = numeric_score("연 1000만원 한도의 15%", "1,000만원;15%")
    assert score == 1.0

def test_numeric_thousand_variant():
    score, _ = numeric_score("총급여액 8000만원 이하", "8천만원")
    assert score == 1.0

def test_numeric_partial():
    score, missing = numeric_score("15%를 공제받습니다.", "15%;17%")
    assert score == 0.5 and missing == ["17%"]

def test_numeric_won_unit_variant():
    # 실제 Gemini 답변에서 관측: "1명당 1,500,000원입니다"
    score, _ = numeric_score("1명당 1,500,000원입니다.", "150만원")
    assert score == 1.0

def test_numeric_percent_fraction_variant():
    # 실제 Gemini 답변에서 관측: "소득세의 100분의 90에 상당하는"
    score, _ = numeric_score("소득세의 100분의 90에 상당하는 세액", "90%")
    assert score == 1.0

def test_numeric_no_keys():
    score, _ = numeric_score("아무 답변", None)
    assert score is None


# ── 조건 포함률 ──
def test_condition_with_synonym():
    score, _ = condition_score("세대의 세대주여야 합니다.", "무주택|세대의 세대주")
    assert score == 1.0

def test_condition_missing():
    score, missing = condition_score("15%를 공제받습니다.", "무주택;세대주")
    assert score == 0.0 and len(missing) == 2


# ── 페이지 인용 ──
def test_cited_pages_extract():
    assert cited_pages("공제받습니다. (근거: p.220, p.221)") == [220, 221]

def test_citation_valid():
    score, bad = citation_score("(근거: p.220)", [220, 35, 25])
    assert score == 1.0 and bad == []

def test_citation_fabricated():
    score, bad = citation_score("(근거: p.999)", [220, 35, 25])
    assert score == 0.0 and bad == [999]

def test_citation_none():
    score, _ = citation_score("인용 없는 답변", [220])
    assert score is None


# ── 오류 코드 태깅 ──
def test_tag_retrieval_fail():
    codes = tag_errors(answerable=True, retrieval_hit=False, full_refusal=True, refusal=True,
                       num_score=None, cond_score=None, cite_score=None)
    assert "E3" in codes and "WR" in codes

def test_tag_numeric_error():
    codes = tag_errors(answerable=True, retrieval_hit=True, full_refusal=False, refusal=False,
                       num_score=0.5, cond_score=1.0, cite_score=1.0)
    assert codes == ["E6"]

def test_tag_no_refusal_on_unanswerable():
    codes = tag_errors(answerable=False, retrieval_hit=None, full_refusal=False, refusal=False,
                       num_score=None, cond_score=None, cite_score=None)
    assert codes == ["E10"]

def test_tag_clean():
    codes = tag_errors(answerable=True, retrieval_hit=True, full_refusal=False, refusal=False,
                       num_score=1.0, cond_score=1.0, cite_score=1.0)
    assert codes == []

def test_tag_partial_hedge_not_wr_but_numeric_error():
    # 핵심 발견 재현: 문구는 있지만(refusal=True) 장문 부분 답변(full_refusal=False)
    # → WR이 아니라 실제로 빠진 숫자만 E6로 잡혀야 한다
    codes = tag_errors(answerable=True, retrieval_hit=True, full_refusal=False, refusal=True,
                       num_score=0.5, cond_score=1.0, cite_score=1.0)
    assert codes == ["E6"]
    assert "WR" not in codes
