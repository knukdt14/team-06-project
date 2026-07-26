"""[담당: 이수홍] 규칙 기반 답변 채점 (LLM/API 불필요).

BERTScore는 문장 구조가 비슷하면 "13만원을 7만원이라 답한 오답"(Q10류)에도
높은 점수를 주므로, 세금 상담에서 치명적인 오류를 별도 규칙으로 채점한다.

지표 (가이드라인 §2):
- 핵심 숫자 정확도  : 정답의 금액·%가 답변에 포함됐는지 (제일 중요)
- 조건 포함률       : 필수 조건(무주택, 세대주 등)이 답변에 포함됐는지
- 페이지 인용 정확도: 답변이 인용한 (근거: p.OO)가 실제 제공된 문맥에서 나왔는지
- 거절 정확도       : 문서 밖 질문을 거절했는지 / 답 있는 질문을 잘못 거절 안 했는지
- 오류 코드 태깅    : 가이드라인 §22의 E1~E11 중 자동 판정 가능한 것

정답 데이터는 references.csv의 추가 컬럼을 사용한다:
- key_numbers   : "15%;17%;1,000만원"  (';'로 구분, 전부 포함해야 만점)
- key_conditions: "무주택;세대주|세대의 세대주"  (';'로 항목, '|'로 동의어)
- answerable    : 1=문서에 답 있음, 0=문서 밖 질문(거절해야 정답)
"""
import re

# ── 거절 판정 ────────────────────────────────────────
REFUSAL_PATTERNS = ("찾을 수 없", "확인할 수 없", "나와 있지 않",
                    "포함되어 있지 않", "문서에 없")


def is_refusal(answer) -> bool:
    """답변이 '문서에서 찾을 수 없습니다'류의 거절인지."""
    return any(p in str(answer) for p in REFUSAL_PATTERNS)


# ── 핵심 숫자 / 조건 포함 채점 ───────────────────────
def _norm(text) -> str:
    """쉼표·공백 제거: '5,500만 원' → '5500만원'."""
    return re.sub(r"[,\s]", "", str(text))


def _variants(key) -> set:
    """숫자 표기 변형 허용.

    8천만원 ↔ 8000만원 / 1000만원 ↔ 1천만원 / 150만원 ↔ 1,500,000원 ↔ 1백5십…
    90% ↔ 100분의 90  (실제 Gemini 답변에서 관측된 표기들)
    """
    k = _norm(key)
    out = {k}
    if "천만" in k:
        out.add(k.replace("천만", "000만"))
    m = re.match(r"^(\d+?)0{3}만", k)
    if m:
        out.add(re.sub(r"^(\d+?)0{3}만", m.group(1) + "천만", k, count=1))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)%", k)
    if m:
        out.add("100분의" + m.group(1))
    m = re.match(r"^(\d+)만원", k)
    if m:
        n = int(m.group(1))
        out.add(f"{n * 10000}원")            # 150만원 → 1500000원
        if n >= 100 and n % 100 == 0:
            out.add(f"{n // 100}백만원")      # 100만원 → 1백만원
    return out


def _split_items(spec):
    if spec is None:
        return []
    s = str(spec).strip()
    if not s or s.lower() == "nan":
        return []
    return [item.strip() for item in s.split(";") if item.strip()]


def coverage(answer, spec, use_variants=False):
    """spec("a;b|c")의 항목 중 답변에 포함된 비율과 누락 목록.

    spec이 비어 있으면 (None, []) — 채점 대상 아님.
    """
    items = _split_items(spec)
    if not items:
        return None, []
    a = _norm(answer)
    missing = []
    for item in items:
        found = False
        for alt in item.split("|"):
            cands = _variants(alt) if use_variants else {_norm(alt)}
            if any(c and c in a for c in cands):
                found = True
                break
        if not found:
            missing.append(item)
    return round(1 - len(missing) / len(items), 3), missing


def numeric_score(answer, key_numbers):
    """핵심 숫자 정확도: (0~1 비율, 누락 목록)."""
    return coverage(answer, key_numbers, use_variants=True)


def condition_score(answer, key_conditions):
    """필수 조건 포함률: (0~1 비율, 누락 목록)."""
    return coverage(answer, key_conditions)


# ── 페이지 인용 채점 ─────────────────────────────────
CITE_RE = re.compile(r"(?:p\.?|페이지)\s*(\d+)")


def cited_pages(answer):
    """답변에서 '(근거: p.220)' 류의 인용 페이지 추출."""
    return [int(m) for m in CITE_RE.findall(str(answer))]


def citation_score(answer, context_pages):
    """인용 페이지가 실제 LLM에게 제공된 문맥 페이지(raw, 0-indexed)에 있는지.

    반환: (비율 or None(인용 없음), 잘못된 인용 목록).
    잘못된 인용 = 문맥에 없는 페이지를 지어냄 (E9 후보).
    """
    cited = cited_pages(answer)
    if not cited:
        return None, []
    ctx = {int(p) for p in context_pages if p is not None and int(p) >= 0}
    bad = [p for p in cited if p not in ctx]
    return round(1 - len(bad) / len(cited), 3), bad


# ── 오류 코드 태깅 (가이드라인 §22) ──────────────────
def tag_errors(*, answerable, retrieval_hit, refusal, num_score, cond_score, cite_score):
    """자동 판정 가능한 오류 코드만 태깅.

    E3  정답 문서 검색 실패          E6  핵심 숫자 누락/오류
    E7  필수 조건 누락               E9  문맥에 없는 페이지 인용
    E10 문서 밖 질문 거절 실패       WR  답 있는 질문을 잘못 거절 (커스텀)
    ※ E1(추출)·E2(청크 경계)·E4(순위)·E5(잡음)·E8(환각)·E11(형식)은 수동 검토.
    """
    codes = []
    if answerable:
        if retrieval_hit is False:
            codes.append("E3")
        if refusal:
            codes.append("WR")
        else:
            if num_score is not None and num_score < 1:
                codes.append("E6")
            if cond_score is not None and cond_score < 1:
                codes.append("E7")
    else:
        if not refusal:
            codes.append("E10")
    if cite_score is not None and cite_score < 1:
        codes.append("E9")
    return codes


# ── DataFrame 일괄 채점 ──────────────────────────────
def score_dataframe(df, context_pages_by_id=None):
    """detail DF에 채점 컬럼을 추가하고 (df, summary dict)를 반환한다.

    필요 컬럼: id, answer, page(정답 물리 페이지), key_numbers, key_conditions.
    answerable 컬럼이 없으면 전부 1로 간주.
    context_pages_by_id: {id: [raw 페이지들]} — 검색된 문맥(0-indexed).
                         없으면 인용 채점은 건너뛴다(None).
    """
    import pandas as pd

    ctx = context_pages_by_id or {}
    rows = []
    for _, r in df.iterrows():
        answerable = bool(int(r.get("answerable", 1) or 0)) if "answerable" in df.columns else True
        refusal = is_refusal(r["answer"])
        n_score, n_miss = numeric_score(r["answer"], r.get("key_numbers"))
        c_score, c_miss = condition_score(r["answer"], r.get("key_conditions"))

        qid = r["id"]
        raw_pages = ctx.get(qid)
        if raw_pages is not None:
            gold = int(r["page"]) if answerable and pd.notna(r.get("page")) else None
            physical = [p + 1 for p in raw_pages if p is not None and p >= 0]
            hit = (gold in physical) if gold is not None else None
            cite, bad_cites = citation_score(r["answer"], raw_pages)
        else:
            hit, cite, bad_cites, physical = None, None, [], None

        codes = tag_errors(answerable=answerable, retrieval_hit=hit, refusal=refusal,
                           num_score=n_score, cond_score=c_score, cite_score=cite)
        rows.append({
            "id": qid, "retrieved_pages": physical, "retrieval_hit": hit,
            "refusal": refusal,
            "numeric_score": n_score, "missing_numbers": ";".join(n_miss),
            "condition_score": c_score, "missing_conditions": ";".join(c_miss),
            "citation_score": cite, "bad_citations": bad_cites,
            "error_codes": ",".join(codes),
        })

    scored = df.merge(pd.DataFrame(rows), on="id")

    def _mean(col):
        s = scored[col].dropna()
        return round(float(s.mean()), 4) if len(s) else None

    answerable_mask = (scored["answerable"].fillna(1).astype(int) == 1
                       if "answerable" in scored.columns
                       else pd.Series(True, index=scored.index))
    n_ans = scored[answerable_mask]
    n_unans = scored[~answerable_mask]

    from collections import Counter
    counts = Counter()
    for c in scored["error_codes"]:
        if c:
            counts.update(c.split(","))

    summary = {
        "numeric_acc": _mean("numeric_score"),
        "numeric_perfect": round(float(
            (n_ans["numeric_score"].dropna() == 1).mean()), 4) if len(n_ans) else None,
        "condition_recall": _mean("condition_score"),
        "citation_acc": _mean("citation_score"),
        "hit_rate": _mean("retrieval_hit"),
        "wrong_refusals": int(n_ans["refusal"].sum()),
        "abstention_acc": round(float(n_unans["refusal"].mean()), 4) if len(n_unans) else None,
        "error_counts": "|".join(f"{k}:{v}" for k, v in sorted(counts.items())),
    }
    return scored, summary
