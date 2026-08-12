"""Product facts come from the catalogue, not from the model.

The prompt shows the model a product_id and a chunk of text. It never shows a name,
a brand or a price, so the model invents them -- a live run returned three different
products all called "Hydrating Serum". The router repairs that from the catalogue.

The subtle half is *which* catalogue. RetrievalResult.products holds only the top-k
summary, while the evidence block spans more products than that, and the model may
recommend any of them. A lookup built from products[] alone leaves those rows with
an invented name, which is exactly what the first attempt did.
"""
import pytest

from app.schemas import Product, UserProfile
from skincare.rag.mock_retrieval import MockRetriever


@pytest.fixture
def retriever():
    return MockRetriever()


def test_product_table_resolves_ids_beyond_the_top_k_summary():
    """The property the router depends on: the lookup is not capped at top_k.

    Two retrievers rather than two calls, because MockRetriever's rng is stateful
    and a second search would not be comparable to the first.
    """
    profile = UserProfile(query="dry skin")
    narrow = MockRetriever().search(profile, None, top_k=1)
    wide = MockRetriever().search(profile, None, top_k=3)

    narrow_ids = {p.product_id for p in narrow.products}
    wide_ids = {p.product_id for p in wide.products}
    beyond = wide_ids - narrow_ids
    assert beyond, "fixture must offer more than one candidate product"

    table = MockRetriever().product_table(wide_ids)
    for pid in beyond:
        assert pid in table, f"{pid} sits outside a top-1 summary and must still resolve"
        assert table[pid].name, "a resolved product must carry a real name"


def test_unknown_ids_are_skipped_rather_than_faked(retriever):
    table = retriever.product_table(["P005", "DEFINITELY-NOT-A-PRODUCT"])
    assert "DEFINITELY-NOT-A-PRODUCT" not in table
    assert isinstance(table.get("P005"), Product)


def test_lookup_is_deduplicated_and_order_independent(retriever):
    once = retriever.product_table(["P005", "P006"])
    twice = retriever.product_table(["P006", "P005", "P005"])
    assert once.keys() == twice.keys()
    assert once["P005"].name == twice["P005"].name


def test_empty_input_gives_an_empty_table(retriever):
    assert retriever.product_table([]) == {}


def test_the_mock_and_real_retriever_expose_the_same_method():
    """Duck typing is the contract here -- deps.py swaps one for the other."""
    from skincare.rag.retrieve import Retriever
    assert callable(getattr(Retriever, "product_table", None))
    assert callable(getattr(MockRetriever, "product_table", None))
