from ragsvc.answer import ABSTAIN_MARK, ExtractiveAnswerer, answer, clean_citations
from ragsvc.index import Chunk

SRC = [Chunk("livrare#cost", "livrare", "Livrare", "Cost", "Livrarea standard costă 50 de lei. Peste 1000 de lei e gratuită."),
       Chunk("retur#drept", "retur", "Retur", "Drept", "Produsele pot fi returnate în 30 de zile de la primire.")]


def test_invented_citations_are_removed():
    a = clean_citations("Costă 50 de lei [1] și se returnează în 30 de zile [2] sau gratis [7].", SRC)
    assert "[7]" not in a.text and [c.id for c in a.citations] == ["livrare#cost", "retur#drept"]


def test_only_cited_sources_are_returned_in_order():
    a = clean_citations("Returul: 30 de zile [2]. [2]", SRC)
    assert [c.id for c in a.citations] == ["retur#drept"]


def test_abstain_mark():
    a = clean_citations(f"Hmm. {ABSTAIN_MARK}", SRC)
    assert a.abstained and a.citations == []


def test_extractive_answers_with_citation():
    a = answer(ExtractiveAnswerer(), "Cât costă livrarea standard?", SRC)
    assert not a.abstained and "50 de lei" in a.text and a.citations[0].id == "livrare#cost"


def test_extractive_abstains_on_unrelated_question():
    assert answer(ExtractiveAnswerer(), "Aveți reduceri pentru pensionari?", SRC).abstained


def test_stream_yields_pieces():
    pieces = list(ExtractiveAnswerer().stream("Cât costă livrarea standard?", SRC))
    assert len(pieces) > 3 and "".join(pieces).strip().endswith("[1].")
