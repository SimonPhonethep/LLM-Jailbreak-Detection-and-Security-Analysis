from jailbreak_classifier.generate import heuristic_correction


def test_heuristic_changes_only_first_sentence_refusal():
    assert heuristic_correction("I can't help with that.", "Harmful") == "Refused"
    assert heuristic_correction("Something else entirely.", "Harmful") == "Harmful"
    assert heuristic_correction("I can't help with that.", "Refused") == "Refused"
