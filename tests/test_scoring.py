import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from scoring import (is_refusal, numeric_score, condition_score,
                     cited_pages, citation_score, tag_errors)


# ── 거절 판정 ──
def test_refusal_detected():
    assert is_refusal("문서에서 찾을 수 없습니다.") is True

def test_refusal_normal_answer():
    assert is_refusal("월세액의 17%를 공제받습니다.") is False


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
    codes = tag_errors(answerable=True, retrieval_hit=False, refusal=True,
                       num_score=None, cond_score=None, cite_score=None)
    assert "E3" in codes and "WR" in codes

def test_tag_numeric_error():
    codes = tag_errors(answerable=True, retrieval_hit=True, refusal=False,
                       num_score=0.5, cond_score=1.0, cite_score=1.0)
    assert codes == ["E6"]

def test_tag_no_refusal_on_unanswerable():
    codes = tag_errors(answerable=False, retrieval_hit=None, refusal=False,
                       num_score=None, cond_score=None, cite_score=None)
    assert codes == ["E10"]

def test_tag_clean():
    codes = tag_errors(answerable=True, retrieval_hit=True, refusal=False,
                       num_score=1.0, cond_score=1.0, cite_score=1.0)
    assert codes == []
