import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from eval_retrieval import page_hit, reciprocal_rank, retrieved_physical_pages


class FakeDoc:
    def __init__(self, metadata):
        self.metadata = metadata


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
