import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from eval_retrieval import (
    count_duplicate_sentences,
    get_chunk_count,
    get_embedding_model_name,
    page_hit,
    reciprocal_rank,
    retrieval_error_code,
    retrieved_physical_pages,
    split_normalized_sentences,
)


class FakeDoc:
    def __init__(self, metadata, page_content=""):
        self.metadata = metadata
        self.page_content = page_content


class FakeCollection:
    def count(self):
        return 1132


class FakeChroma:
    _collection = FakeCollection()


class FakeIndex:
    ntotal = 635


class FakeFaiss:
    index = FakeIndex()


class FakeConfig:
    HF_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
    GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def test_physical_pages_add_one():
    docs = [FakeDoc({"page": 219}), FakeDoc({"page": 5})]
    assert retrieved_physical_pages(docs) == [220, 6]

def test_physical_pages_missing():
    assert retrieved_physical_pages([FakeDoc({})]) == [-1]

def test_page_hit_true():
    assert page_hit([220, 6, 5], 220) is True

def test_page_hit_false():
    assert page_hit([6, 5], 220) is False

def test_reciprocal_rank_first():
    assert reciprocal_rank([220, 6], 220) == 1.0

def test_reciprocal_rank_second():
    assert reciprocal_rank([6, 220], 220) == 0.5

def test_reciprocal_rank_miss():
    assert reciprocal_rank([6, 5], 220) == 0.0


# ── 복수 정답 페이지 (요약;상세) ──
from eval_retrieval import parse_gold_pages

def test_parse_gold_pages_multi():
    assert parse_gold_pages("6;182") == [6, 182]

def test_parse_gold_pages_single_int():
    assert parse_gold_pages(220) == [220]

def test_page_hit_multi_gold_detail_page():
    # 요약(6)은 못 찾았지만 상세(182)를 찾음 → hit
    assert page_hit([182, 20, 203], [6, 182]) is True

def test_page_hit_multi_gold_miss():
    assert page_hit([10, 164, 162], [8, 188]) is False

def test_reciprocal_rank_multi_gold_best():
    # 상세(182)가 1위 → 1.0
    assert reciprocal_rank([182, 6, 5], [6, 182]) == 1.0


def test_retrieval_error_code():
    assert retrieval_error_code(True) == ""
    assert retrieval_error_code(False) == "E3"


def test_embedding_model_name():
    assert get_embedding_model_name("huggingface", FakeConfig) == "jhgan/ko-sroberta-multitask"


def test_chunk_count_chroma():
    assert get_chunk_count(FakeChroma()) == 1132


def test_chunk_count_faiss():
    assert get_chunk_count(FakeFaiss()) == 635


def test_split_normalized_sentences():
    text = "첫 번째  문장입니다.\n두 번째 문장입니다!"
    assert split_normalized_sentences(text) == ["첫 번째 문장입니다.", "두 번째 문장입니다!"]


def test_count_duplicate_sentences_counts_extra_occurrences():
    docs = [
        FakeDoc({}, "공통 문장입니다.\n첫 번째 청크입니다."),
        FakeDoc({}, "공통   문장입니다.\n두 번째 청크입니다."),
        FakeDoc({}, "공통 문장입니다."),
    ]
    assert count_duplicate_sentences(docs) == 2


def test_count_duplicate_sentences_without_duplicates():
    docs = [
        FakeDoc({}, "첫 번째 문장입니다."),
        FakeDoc({}, "두 번째 문장입니다."),
    ]
    assert count_duplicate_sentences(docs) == 0
