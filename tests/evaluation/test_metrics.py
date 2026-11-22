import math
import os

import pytest

from clipcap.evaluation import metrics as metrics_module
from clipcap.evaluation.metrics import (
    CaptionScores,
    compute_metrics,
    meteor_available,
    tokenize_for_scoring,
)

REFERENCES = {
    "dog.jpg": ["A dog runs in the park.", "The dog is running outside."],
    "cat.jpg": ["A cat sleeps on a sofa.", "A cat is sleeping."],
    "kids.jpg": ["Two children play football.", "Kids are playing with a ball."],
}
PERFECT = {
    "dog.jpg": "a dog runs in the park",
    "cat.jpg": "a cat sleeps on a sofa",
    "kids.jpg": "two children play football",
}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A dog runs.", "a dog runs"),
        ("  A  Dog ,  running !  ", "a dog running"),
        ("The man's hat", "the man's hat"),
        ("A black-and-white dog", "a black and white dog"),
        ("snake_case", "snake case"),
        ("...", ""),
    ],
)
def test_tokenize_for_scoring(text: str, expected: str) -> None:
    assert tokenize_for_scoring(text) == expected


def test_perfect_captions_get_maximal_bleu_and_rouge() -> None:
    scores = compute_metrics(REFERENCES, PERFECT)
    for value in (scores.bleu1, scores.bleu2, scores.bleu3, scores.bleu4, scores.rouge_l):
        assert value == pytest.approx(1.0, abs=1e-6)
    assert scores.cider > 1.0
    assert scores.meteor is None


def test_scores_ignore_case_and_punctuation() -> None:
    shouted = {image: caption.upper() + " !!!" for image, caption in PERFECT.items()}
    assert compute_metrics(REFERENCES, shouted) == compute_metrics(REFERENCES, PERFECT)


def test_bleu1_matches_a_hand_computed_value() -> None:
    # Unigram precision is 3/3; the hypothesis has 3 tokens against a 6-token
    # reference, so the brevity penalty is exp(1 - 6/3).
    scores = compute_metrics({"x": ["a dog runs in the park"]}, {"x": "a dog runs"})
    assert scores.bleu1 == pytest.approx(math.exp(-1.0), abs=1e-6)


def test_bleu_precision_is_clipped() -> None:
    # "the" appears twice at most in the reference: clipped precision is 2/4.
    scores = compute_metrics({"x": ["the cat the dog"]}, {"x": "the the the the"})
    assert scores.bleu1 == pytest.approx(0.5, abs=1e-6)
    assert scores.bleu2 == pytest.approx(0.0, abs=1e-4)


def test_rouge_l_matches_a_hand_computed_value() -> None:
    # LCS("a cat sleeps", "a cat is sleeping") = "a cat": P = 2/3, R = 2/4, beta = 1.2.
    precision, recall, beta = 2 / 3, 2 / 4, 1.2
    expected = (1 + beta**2) * precision * recall / (recall + beta**2 * precision)
    scores = compute_metrics({"x": ["a cat is sleeping"]}, {"x": "a cat sleeps"})
    assert scores.rouge_l == pytest.approx(expected, abs=1e-6)


def test_bleu_uses_the_best_matching_reference() -> None:
    single = compute_metrics({"x": ["a completely different sentence"]}, {"x": "a dog runs"})
    multiple = compute_metrics(
        {"x": ["a completely different sentence", "a dog runs"]}, {"x": "a dog runs"}
    )
    assert multiple.bleu1 == pytest.approx(1.0, abs=1e-6)
    assert single.bleu1 < 0.5


def test_worse_captions_score_lower_on_every_metric() -> None:
    swapped = {
        "dog.jpg": PERFECT["cat.jpg"],
        "cat.jpg": PERFECT["kids.jpg"],
        "kids.jpg": PERFECT["dog.jpg"],
    }
    good = compute_metrics(REFERENCES, PERFECT).to_dict()
    bad = compute_metrics(REFERENCES, swapped).to_dict()
    assert good.keys() == bad.keys() == {"bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "cider"}
    for name, value in good.items():
        assert bad[name] < value, name


def test_unrelated_captions_score_zero() -> None:
    unrelated = dict.fromkeys(REFERENCES, "xyzzy plugh")
    scores = compute_metrics(REFERENCES, unrelated)
    assert scores.bleu1 == pytest.approx(0.0, abs=1e-6)
    assert scores.rouge_l == pytest.approx(0.0, abs=1e-6)
    assert scores.cider == pytest.approx(0.0, abs=1e-6)


def test_empty_hypothesis_is_scored_as_zero_for_its_image() -> None:
    with_empty = {**PERFECT, "kids.jpg": ""}
    scores = compute_metrics(REFERENCES, with_empty)
    perfect = compute_metrics(REFERENCES, PERFECT)
    assert 0.0 < scores.rouge_l < perfect.rouge_l
    assert 0.0 < scores.cider < perfect.cider
    assert all(math.isfinite(value) for value in scores.to_dict().values())


def test_result_does_not_depend_on_image_order() -> None:
    reversed_references = dict(reversed(list(REFERENCES.items())))
    hypotheses = {**PERFECT, "cat.jpg": "a cat"}
    assert compute_metrics(reversed_references, hypotheses) == compute_metrics(
        REFERENCES, hypotheses
    )


@pytest.mark.parametrize(
    ("references", "hypotheses", "message"),
    [
        (REFERENCES, {}, "no caption to score"),
        (REFERENCES, {"dog.jpg": "a dog"}, "2 without hypothesis"),
        ({"dog.jpg": ["a dog"]}, {"dog.jpg": "a dog", "z.jpg": "a cat"}, "1 without reference"),
        ({"dog.jpg": []}, {"dog.jpg": "a dog"}, "no usable reference"),
        ({"dog.jpg": ["...", " "]}, {"dog.jpg": "a dog"}, "no usable reference"),
    ],
)
def test_inconsistent_inputs_are_rejected(
    references: dict[str, list[str]], hypotheses: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_metrics(references, hypotheses)


def test_meteor_without_java_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_module.shutil, "which", lambda name: None)
    assert meteor_available() is False
    with pytest.raises(RuntimeError, match="requires a Java runtime"):
        compute_metrics(REFERENCES, PERFECT, meteor=True)


@pytest.mark.java
@pytest.mark.skipif(not meteor_available(), reason="METEOR needs a Java runtime")
def test_meteor_is_computed_when_java_is_available() -> None:
    perfect = compute_metrics(REFERENCES, PERFECT, meteor=True)
    worse = compute_metrics(REFERENCES, dict.fromkeys(REFERENCES, "a man"), meteor=True)
    assert perfect.meteor is not None
    assert worse.meteor is not None
    assert 0.0 <= worse.meteor < perfect.meteor <= 1.0
    assert "meteor" in perfect.to_dict()


@pytest.mark.java
@pytest.mark.skipif(not meteor_available(), reason="METEOR needs a Java runtime")
@pytest.mark.parametrize("locale", ["fr_FR.UTF-8", "C"])
def test_meteor_does_not_depend_on_the_system_locale(
    monkeypatch: pytest.MonkeyPatch, locale: str
) -> None:
    monkeypatch.setenv("LC_ALL", locale)
    monkeypatch.setenv("LANG", locale)
    assert compute_metrics(REFERENCES, PERFECT, meteor=True).meteor == pytest.approx(1.0, abs=0.05)
    assert os.environ["LC_ALL"] == locale


@pytest.mark.java
@pytest.mark.skipif(not meteor_available(), reason="METEOR needs a Java runtime")
def test_meteor_failure_is_reported_without_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    from pycocoevalcap.meteor.meteor import Meteor

    def crash(self: Meteor, gts: object, res: object) -> None:
        self.lock.acquire()  # the library leaves its lock held when scoring fails
        raise ValueError("could not convert string to float")

    monkeypatch.setattr(Meteor, "compute_score", crash)
    monkeypatch.delenv("LC_ALL", raising=False)
    with pytest.raises(RuntimeError, match="METEOR failed"):
        compute_metrics(REFERENCES, PERFECT, meteor=True)
    assert "LC_ALL" not in os.environ


def test_scores_serialise_without_missing_metrics() -> None:
    scores = CaptionScores(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    assert scores.to_dict() == {
        "bleu1": 0.1,
        "bleu2": 0.2,
        "bleu3": 0.3,
        "bleu4": 0.4,
        "rouge_l": 0.5,
        "cider": 0.6,
    }
