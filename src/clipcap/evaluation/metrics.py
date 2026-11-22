"""Corpus-level caption metrics: BLEU-1..4, ROUGE-L, CIDEr and optionally METEOR.

Scores come from ``pycocoevalcap``, the reference implementation behind the
figures reported in the captioning literature. The only deviation is
tokenisation: the official pipeline calls the Java PTB tokenizer, whereas
:func:`tokenize_for_scoring` is a pure-Python equivalent for plain English
captions (lower-casing, punctuation removal), so that BLEU, ROUGE-L and CIDEr
need no Java runtime. METEOR still does and is therefore optional.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass

_PUNCTUATION = re.compile(r"[^\w\s']|_")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class CaptionScores:
    """Metric values in their native ranges.

    BLEU, ROUGE-L and METEOR lie in [0, 1]; CIDEr lies in [0, 10]. Papers
    usually report all of them multiplied by 100.
    """

    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    rouge_l: float
    cider: float
    meteor: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Return the available scores as a dictionary."""
        return {name: value for name, value in asdict(self).items() if value is not None}


def tokenize_for_scoring(text: str) -> str:
    """Lower-case ``text``, drop punctuation and collapse whitespace."""
    text = _PUNCTUATION.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", text).strip()


def meteor_available() -> bool:
    """Tell whether METEOR can run, i.e. whether a Java runtime is installed."""
    return shutil.which("java") is not None


def compute_metrics(
    references: Mapping[str, Sequence[str]],
    hypotheses: Mapping[str, str],
    *,
    meteor: bool = False,
) -> CaptionScores:
    """Score one generated caption per image against its reference captions.

    ``references`` and ``hypotheses`` must describe the same images. An empty
    hypothesis is legal and simply scores zero for its image.
    """
    if not hypotheses:
        raise ValueError("no caption to score")
    if set(references) != set(hypotheses):
        only_references = sorted(set(references) - set(hypotheses))
        only_hypotheses = sorted(set(hypotheses) - set(references))
        raise ValueError(
            "references and hypotheses must cover the same images: "
            f"{len(only_references)} without hypothesis {only_references[:3]}, "
            f"{len(only_hypotheses)} without reference {only_hypotheses[:3]}"
        )
    if meteor and not meteor_available():
        raise RuntimeError("METEOR requires a Java runtime, none was found on PATH")

    images = sorted(hypotheses)
    ground_truth: dict[str, list[str]] = {}
    for image in images:
        tokenized = [tokenize_for_scoring(reference) for reference in references[image]]
        tokenized = [reference for reference in tokenized if reference]
        if not tokenized:
            raise ValueError(f"image '{image}' has no usable reference caption")
        ground_truth[image] = tokenized
    results = {image: [tokenize_for_scoring(hypotheses[image])] for image in images}

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    bleu, _ = Bleu(4).compute_score(ground_truth, results, verbose=0)
    rouge, _ = Rouge().compute_score(ground_truth, results)
    cider, _ = Cider().compute_score(ground_truth, results)
    meteor_score = _meteor(ground_truth, results) if meteor else None
    return CaptionScores(
        bleu1=float(bleu[0]),
        bleu2=float(bleu[1]),
        bleu3=float(bleu[2]),
        bleu4=float(bleu[3]),
        rouge_l=float(rouge),
        cider=float(cider),
        meteor=meteor_score,
    )


@contextmanager
def _c_locale() -> Iterator[None]:
    """Temporarily force the C locale for subprocesses started in the block."""
    saved = {name: os.environ.get(name) for name in ("LC_ALL", "LANG")}
    os.environ.update(LC_ALL="C", LANG="C")
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value


def _meteor(ground_truth: dict[str, list[str]], results: dict[str, list[str]]) -> float:
    from pycocoevalcap.meteor.meteor import Meteor

    # METEOR's Java process parses decimals with the system locale and crashes
    # under locales that use a decimal comma (French, German, ...).
    with _c_locale():
        scorer = Meteor()
    try:
        score, _ = scorer.compute_score(ground_truth, results)
    except Exception as exc:
        raise RuntimeError(f"METEOR failed: {exc!r}") from exc
    finally:
        # The scorer keeps a Java subprocess alive and, when scoring fails,
        # leaves its lock held, which would deadlock its destructor.
        if scorer.lock.locked():
            scorer.lock.release()
        scorer.meteor_p.kill()
        scorer.meteor_p.wait()
        for stream in (scorer.meteor_p.stdin, scorer.meteor_p.stdout, scorer.meteor_p.stderr):
            stream.close()
    return float(score)
