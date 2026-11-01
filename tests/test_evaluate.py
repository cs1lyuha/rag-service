from ragsvc.answer import ExtractiveAnswerer
from ragsvc.evaluate import check, evaluate, load_index


def test_quality_gates_pass_on_current_version():
    m = evaluate(load_index(), ExtractiveAnswerer())
    assert m["questions"] == 65 and check(m) == []


def test_gate_catches_a_regression():
    broken = ExtractiveAnswerer(threshold=0.9)  # refuză aproape tot
    assert any("false_abstain" in f for f in check(evaluate(load_index(), broken)))
