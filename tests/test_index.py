import pytest

from ragsvc.index import HybridIndex, split_sections


def test_sections_and_ids():
    chunks = split_sections("livrare", "# Livrare\n\n## Costul livrării\nCostă 50 de lei.\n\n## Termene\n1-3 zile.\n")
    assert [c.id for c in chunks] == ["livrare#costul-livrarii", "livrare#termene"]
    assert chunks[0].title == "Livrare"
    assert split_sections("x", "# Doar titlu\ntext liber")[0].id == "x#intro"


def test_kb_search(kb_index):
    assert len(kb_index.docs) == 12
    top = kb_index.search("Cât costă livrarea standard?", k=3)
    assert top[0][0].id == "livrare#costul-livrarii"


def test_paraphrase_search(kb_index):
    top = kb_index.search("Mi-a căzut telefonul în apă, intră în garanție?", k=3)
    assert any(c.doc == "garantie" for c, _, _ in top)


def test_upsert_is_incremental_and_idempotent():
    idx = HybridIndex()
    assert idx.upsert("faq", "# FAQ\n\n## Program\nLuni-vineri 9-18.") == "created"
    assert idx.upsert("faq", "# FAQ\n\n## Program\nLuni-vineri 9-18.") == "unchanged"
    assert idx.upsert("faq", "# FAQ\n\n## Program\nNon-stop.") == "updated"
    assert [c.text for c in idx.chunks] == ["Non-stop."]
    assert idx.search("program", k=1)[0][0].text == "Non-stop."


def test_delete_and_empty_search():
    idx = HybridIndex()
    idx.upsert("a", "# A\n\n## S\ntext")
    assert idx.delete("a") and not idx.delete("a")
    assert idx.search("text") == []


@pytest.mark.parametrize("bad", ["../etc", "Mare", "", "a" * 65, "cu spatiu"])
def test_document_names_are_validated(bad):
    with pytest.raises(ValueError):
        HybridIndex().upsert(bad, "# x")
