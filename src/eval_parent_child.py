"""Parent-child retrieval 독립 실험.

마크다운 제목을 경계로 여러 페이지에 걸친 parent 절을 만들고, parent 안의
작은 child 청크를 검색한 뒤 상위 3개 고유 parent의 전체 페이지를 반환한다.

기존 검색·설정·평가 파일은 수정하지 않는다. 임베딩은 config 기본값인
BAAI/bge-m3만 사용한다.

실행 예:
    python src/eval_parent_child.py --rebuild
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from build_vectorstore import get_embeddings
from eval_retrieval import (
    UNANSWERABLE_MARKER,
    parse_gold_pages,
    score_retrieval_result,
)
from load_pdf import load_documents_markdown


CHILD_SIZE = 250
CHILD_OVERLAP = 50
CHILD_TOP_N = 15
PARENT_TOP_K = 3
EXPECTED_MODEL = "BAAI/bge-m3"
HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s+\S")


@dataclass
class ParentSection:
    """제목 하나에 속한 parent 절과 원문 문자 구간별 페이지 정보."""

    parent_id: str
    title: str
    text: str
    pages: list[int]
    page_spans: list[tuple[int, int, int]]


def _make_parent(index, title, pieces):
    """(줄, page) 조각을 ParentSection으로 변환한다."""

    while pieces and not pieces[0][0].strip():
        pieces.pop(0)
    while pieces and not pieces[-1][0].strip():
        pieces.pop()
    if not pieces:
        return None

    text_parts = []
    page_spans = []
    cursor = 0
    for line_index, (line, page) in enumerate(pieces):
        if line_index:
            text_parts.append("\n")
            cursor += 1
        start = cursor
        text_parts.append(line)
        cursor += len(line)
        page_spans.append((start, cursor, page))

    return ParentSection(
        parent_id=f"parent_{index:04d}",
        title=title,
        text="".join(text_parts),
        pages=sorted({page for _, _, page in page_spans}),
        page_spans=page_spans,
    )


def build_parent_sections(documents):
    """페이지를 잇고 마크다운 제목 줄을 경계로 parent 절을 만든다."""

    parents = []
    current_title = "(문서 시작)"
    current_pieces = []

    def finish_current():
        nonlocal current_pieces
        parent = _make_parent(
            len(parents), current_title, current_pieces
        )
        if parent is not None:
            parents.append(parent)
        current_pieces = []

    for document in documents:
        page = int(document.metadata["page"])
        for line in document.page_content.splitlines():
            if HEADING_PATTERN.match(line):
                finish_current()
                current_title = line.strip()
            current_pieces.append((line, page))

    finish_current()
    return parents


def _pages_for_span(parent, start, end):
    """parent 문자 범위와 겹치는 물리 페이지(0-indexed)를 반환한다."""

    pages = {
        page
        for span_start, span_end, page in parent.page_spans
        if span_start < end and span_end > start
    }
    return sorted(pages or set(parent.pages[:1]))


def build_child_documents(
    parents,
    child_size=CHILD_SIZE,
    child_overlap=CHILD_OVERLAP,
):
    """parent별 text를 작은 child로 나누고 parent 연결 메타데이터를 붙인다."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    children = []

    for parent in parents:
        source = Document(
            page_content=parent.text,
            metadata={"parent_id": parent.parent_id},
        )
        for local_index, child in enumerate(splitter.split_documents([source])):
            start = int(child.metadata.get("start_index", 0))
            end = start + len(child.page_content)
            child_pages = _pages_for_span(parent, start, end)

            # Chroma 메타데이터에는 list를 직접 저장할 수 없어 JSON 문자열로 보존한다.
            child.metadata = {
                "child_id": f"{parent.parent_id}_child_{local_index:04d}",
                "parent_id": parent.parent_id,
                "parent_pages": json.dumps(parent.pages),
                "child_pages": json.dumps(child_pages),
                "page": child_pages[0],
                "start_index": start,
            }
            children.append(child)

    return children


def _artifact_path(child_size, child_overlap):
    model_tag = config.HF_EMBEDDING_MODEL.split("/")[-1]
    return (
        config.ARTIFACTS_DIR
        / f"parent_child_huggingface_{model_tag}_cs{child_size}_ov{child_overlap}_md"
    )


def _manifest_payload(parents, children, child_size, child_overlap):
    return {
        "embedding_model": config.HF_EMBEDDING_MODEL,
        "loader": "markdown",
        "child_size": child_size,
        "child_overlap": child_overlap,
        "parent_count": len(parents),
        "child_count": len(children),
    }


def build_or_load_index(
    parents,
    children,
    child_size=CHILD_SIZE,
    child_overlap=CHILD_OVERLAP,
    rebuild=False,
):
    """별도 경로의 parent-child Chroma 인덱스를 만들거나 검증 후 로드한다."""

    from langchain_chroma import Chroma

    path = _artifact_path(child_size, child_overlap)
    manifest_path = path / "parent_child_manifest.json"
    expected = _manifest_payload(
        parents, children, child_size, child_overlap
    )

    if rebuild and path.exists():
        resolved = path.resolve()
        if (
            resolved.parent != config.ARTIFACTS_DIR.resolve()
            or not resolved.name.startswith("parent_child_")
        ):
            raise ValueError(f"삭제 안전검사 실패: {resolved}")
        shutil.rmtree(resolved)

    embeddings = get_embeddings()
    if path.exists():
        if not manifest_path.exists():
            raise RuntimeError(
                f"기존 인덱스에 manifest가 없습니다. --rebuild 필요: {path}"
            )
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(
                "기존 인덱스 설정이 현재 실험과 다릅니다. --rebuild로 다시 만드세요.\n"
                f"기존={actual}\n현재={expected}"
            )
        print(f"[parent-child] 기존 인덱스 로드: {path.name}")
        return Chroma(
            persist_directory=str(path),
            embedding_function=embeddings,
        ), path

    print(f"[parent-child] child {len(children)}개 임베딩 시작")
    vectorstore = Chroma.from_documents(
        children,
        embeddings,
        persist_directory=str(path),
    )
    manifest_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[parent-child] 인덱스 저장 완료: {path}")
    return vectorstore, path


def load_evalset():
    """기존 26문항과 신규 40문항을 메모리에서만 합쳐 66문항을 만든다."""

    def append_missing(base_path, extra_path):
        base = pd.read_csv(base_path, encoding="utf-8-sig")
        if extra_path.exists():
            extra = pd.read_csv(extra_path, encoding="utf-8-sig")
            extra = extra[~extra["id"].isin(base["id"])]
            base = pd.concat([base, extra], ignore_index=True)
        return base

    questions = append_missing(
        config.EVAL_DIR / "questions.csv",
        config.EVAL_DIR / "questions_new2.csv",
    )
    references = append_missing(
        config.EVAL_DIR / "references.csv",
        config.EVAL_DIR / "references_new2.csv",
    )

    if "qtype" not in questions.columns:
        questions["qtype"] = "기존26"
    else:
        questions["qtype"] = questions["qtype"].fillna("기존26")

    result = questions.merge(
        references,
        on="id",
        how="inner",
        validate="one_to_one",
    ).sort_values("id")
    expected_ids = set(range(1, 67))
    actual_ids = set(result["id"].astype(int))
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"66문항 평가셋 ID 불일치: 누락={sorted(expected_ids - actual_ids)}, "
            f"추가={sorted(actual_ids - expected_ids)}"
        )
    return result


def retrieve_parents(vectorstore, parent_by_id, question, top_n, parent_k):
    """child top-N을 처음 등장한 parent 기준으로 묶어 top-k parent를 반환한다."""

    docs = vectorstore.similarity_search(question, k=top_n)
    selected = []
    seen = set()
    for child_rank, document in enumerate(docs, start=1):
        parent_id = document.metadata["parent_id"]
        if parent_id in seen:
            continue
        seen.add(parent_id)
        parent = parent_by_id[parent_id]
        selected.append(
            {
                "parent_id": parent_id,
                "child_rank": child_rank,
                "pages": [page + 1 for page in parent.pages],
            }
        )
        if len(selected) == parent_k:
            break

    retrieved_pages = sorted(
        {page for parent in selected for page in parent["pages"]}
    )
    return selected, retrieved_pages


def parent_reciprocal_rank(selected_parents, gold_pages):
    """정답 페이지를 포함한 첫 parent 순위의 역수를 반환한다."""

    golds = set(gold_pages)
    for rank, parent in enumerate(selected_parents, start=1):
        if golds.intersection(parent["pages"]):
            return round(1.0 / rank, 3)
    return 0.0


def _as_bool(value):
    return str(value).strip().lower() == "true"


def load_flat_baseline():
    path = config.EVAL_DIR / "retrieval_final66_pypdf.csv"
    if not path.exists():
        raise FileNotFoundError(f"비교 기준 결과가 없습니다: {path}")
    baseline = pd.read_csv(path, encoding="utf-8-sig")
    if len(baseline) != 66:
        raise RuntimeError(f"PyPDF 비교 기준이 66행이 아닙니다: {len(baseline)}")
    return baseline.set_index("id")


def evaluate(
    evalset,
    vectorstore,
    parents,
    child_count,
    top_n=CHILD_TOP_N,
    parent_k=PARENT_TOP_K,
):
    parent_by_id = {parent.parent_id: parent for parent in parents}
    baseline = load_flat_baseline()
    rows = []

    for number, (_, row) in enumerate(evalset.iterrows(), start=1):
        selected, retrieved_pages = retrieve_parents(
            vectorstore,
            parent_by_id,
            row["question"],
            top_n,
            parent_k,
        )
        answerable = int(row.get("answerable", 1))
        gold_pages = parse_gold_pages(row.get("page"))
        score = score_retrieval_result(
            retrieved_pages,
            row.get("page"),
            answerable,
        )

        if answerable:
            reciprocal_rank = parent_reciprocal_rank(selected, gold_pages)
            all_pages_covered = set(gold_pages).issubset(retrieved_pages)
        else:
            reciprocal_rank = UNANSWERABLE_MARKER
            all_pages_covered = UNANSWERABLE_MARKER

        flat = baseline.loc[int(row["id"])]
        flat_hit = (
            _as_bool(flat["hit"])
            if answerable
            else UNANSWERABLE_MARKER
        )
        if answerable and score["hit"] != flat_hit:
            hit_change = "IMPROVED" if score["hit"] else "WORSENED"
        elif answerable:
            hit_change = "SAME"
        else:
            hit_change = UNANSWERABLE_MARKER

        rows.append(
            {
                "id": int(row["id"]),
                "question": row["question"],
                "category": row.get("category", ""),
                "qtype": row.get("qtype", "기존26"),
                "embedding_model": config.HF_EMBEDDING_MODEL,
                "parent_count": len(parents),
                "child_count": child_count,
                "child_top_n": top_n,
                "parent_top_k": parent_k,
                "answerable": answerable,
                "gold_page": json.dumps(gold_pages, ensure_ascii=False),
                "retrieved_parent_ids": json.dumps(
                    [parent["parent_id"] for parent in selected]
                ),
                "retrieved_parent_pages": json.dumps(
                    [parent["pages"] for parent in selected]
                ),
                "retrieved_pages": json.dumps(retrieved_pages),
                "returned_page_count": len(retrieved_pages),
                "hit": score["hit"],
                "reciprocal_rank": reciprocal_rank,
                "all_pages_covered": all_pages_covered,
                "error_code": score["error_code"],
                "flat_pypdf_hit": flat_hit,
                "flat_pypdf_retrieved_pages": flat["retrieved_pages"],
                "hit_change_vs_flat_pypdf": hit_change,
            }
        )
        print(
            f"[{number:02d}/66] Q{int(row['id']):02d} "
            f"parents={len(selected)} pages={len(retrieved_pages)} "
            f"hit={score['hit']}"
        )

    return pd.DataFrame(rows)


def print_summary(result, parents, output_path):
    scored = result[result["answerable"] == 1].copy()
    scored["hit_numeric"] = scored["hit"].astype(bool).astype(int)
    scored["rr_numeric"] = pd.to_numeric(scored["reciprocal_rank"])

    hit = scored["hit_numeric"].mean()
    mrr = scored["rr_numeric"].mean()
    average_pages = result["returned_page_count"].mean()
    multi = scored[scored["qtype"] == "멀티페이지"]
    multi_coverage = (
        multi["all_pages_covered"].astype(bool).mean()
        if not multi.empty
        else float("nan")
    )
    huge = [parent for parent in parents if len(parent.pages) >= 10]

    print("\n===== Parent-child 검색 채점 =====")
    print(f"Hit@3: {hit:.3f} | MRR: {mrr:.3f}")
    print(
        "멀티페이지 전체 커버리지: "
        f"{int(multi['all_pages_covered'].astype(bool).sum())}/{len(multi)} "
        f"({multi_coverage:.3f})"
    )
    print(f"평균 반환 페이지 수: {average_pages:.2f}")
    print(
        f"parent {len(parents)}개 | "
        f"10쪽 이상 parent {len(huge)}개 | 결과: {output_path}"
    )

    for qtype, group in scored.groupby("qtype", sort=False):
        print(
            f"  {qtype}: Hit@3={group['hit_numeric'].mean():.3f}, "
            f"MRR={group['rr_numeric'].mean():.3f}, n={len(group)}"
        )

    changed = result[
        result["hit_change_vs_flat_pypdf"].isin(["IMPROVED", "WORSENED"])
    ]
    print("\n[PyPDF flat 대비 Hit 변화]")
    if changed.empty:
        print("  변화 없음")
    else:
        for _, row in changed.iterrows():
            print(
                f"  Q{row['id']}: {row['hit_change_vs_flat_pypdf']} "
                f"| parent-child={row['retrieved_pages']} "
                f"| flat={row['flat_pypdf_retrieved_pages']}"
            )

    if huge:
        print("\n[10쪽 이상 과대 parent]")
        for parent in huge:
            physical_pages = [page + 1 for page in parent.pages]
            print(
                f"  {parent.parent_id}: {len(parent.pages)}쪽 "
                f"{physical_pages} | {parent.title[:70]}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="마크다운 parent-child retrieval 66문항 평가"
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--child-size", type=int, default=CHILD_SIZE)
    parser.add_argument("--child-overlap", type=int, default=CHILD_OVERLAP)
    parser.add_argument("--child-top-n", type=int, default=CHILD_TOP_N)
    parser.add_argument("--parent-top-k", type=int, default=PARENT_TOP_K)
    args = parser.parse_args()

    if config.EMBEDDING_PROVIDER != "huggingface":
        raise RuntimeError(
            f"임베딩 provider가 최종 설정이 아닙니다: {config.EMBEDDING_PROVIDER}"
        )
    if config.HF_EMBEDDING_MODEL != EXPECTED_MODEL:
        raise RuntimeError(
            f"임베딩 모델이 최종 설정이 아닙니다: {config.HF_EMBEDDING_MODEL}"
        )
    if args.child_overlap >= args.child_size:
        raise ValueError("child overlap은 child size보다 작아야 합니다.")

    print(f"[parent-child] embedding={config.HF_EMBEDDING_MODEL}")
    documents = load_documents_markdown()
    parents = build_parent_sections(documents)
    children = build_child_documents(
        parents,
        args.child_size,
        args.child_overlap,
    )
    print(
        f"[parent-child] parent={len(parents)}, child={len(children)}, "
        f"child_size={args.child_size}, overlap={args.child_overlap}"
    )

    vectorstore, index_path = build_or_load_index(
        parents,
        children,
        args.child_size,
        args.child_overlap,
        args.rebuild,
    )
    evalset = load_evalset()
    result = evaluate(
        evalset,
        vectorstore,
        parents,
        len(children),
        args.child_top_n,
        args.parent_top_k,
    )
    output_path = config.EVAL_DIR / "retrieval_parent_child.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print_summary(result, parents, output_path)
    print(f"[parent-child] index={index_path}")


if __name__ == "__main__":
    main()
