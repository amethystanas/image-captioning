import json
from pathlib import Path

import pytest
import torch

from clipcap.config import GenerationConfig
from clipcap.data.dataset import CaptionDataset
from clipcap.data.features import FeatureSet
from clipcap.evaluation.evaluate import evaluate_model, generate_captions
from clipcap.models.clipcap import ClipCaptionModel
from tests.helpers import WordTokenizer, make_features, make_tiny_model

NAMES = ("dog.jpg", "cat.jpg", "kids.jpg", "bird.jpg", "horse.jpg")
REFERENCES = {
    "dog.jpg": ["a dog runs in the park"],
    "cat.jpg": ["a cat sleeps"],
    "kids.jpg": ["two kids play football outside"],
    "bird.jpg": ["a bird flies"],
    "horse.jpg": ["the horse jumps over a fence"],
}
GREEDY = GenerationConfig(strategy="greedy", max_new_tokens=10)
Setup = tuple[ClipCaptionModel, FeatureSet, WordTokenizer]


@pytest.fixture(scope="module")
def memorised() -> Setup:
    """A tiny model trained until it reproduces the reference caption of each image."""
    torch.manual_seed(0)
    features = make_features(NAMES)
    tokenizer = WordTokenizer()
    dataset = CaptionDataset(features, REFERENCES, NAMES, tokenizer, eos_token_id=0)
    batch = dataset.collate([dataset[index] for index in range(len(dataset))])
    model = make_tiny_model(freeze_gpt2=False).train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    for _ in range(150):
        optimizer.zero_grad()
        model(batch.embeddings, batch.tokens, batch.mask).loss.backward()
        optimizer.step()
    return model.eval(), features, tokenizer


def test_generated_captions_are_decoded_text_in_request_order(memorised: Setup) -> None:
    model, features, tokenizer = memorised
    names = ("kids.jpg", "dog.jpg")
    captions = generate_captions(model, features, names, tokenizer, GREEDY, eos_token_id=0)
    assert list(captions) == list(names)
    assert captions == {name: REFERENCES[name][0] for name in names}


@pytest.mark.parametrize("strategy", ["greedy", "beam"])
def test_memorised_captions_reach_perfect_scores(memorised: Setup, strategy: str) -> None:
    model, features, tokenizer = memorised
    generation = GenerationConfig(strategy=strategy, max_new_tokens=10, beam_size=3)
    result = evaluate_model(
        model, features, REFERENCES, NAMES, tokenizer, generation, eos_token_id=0, split="val"
    )
    assert result.split == "val"
    assert result.num_images == 5
    assert result.scores.bleu4 == pytest.approx(1.0, abs=1e-6)
    assert result.scores.rouge_l == pytest.approx(1.0, abs=1e-6)
    assert result.scores.meteor is None


def test_untrained_model_scores_lower_than_trained(memorised: Setup) -> None:
    model, features, tokenizer = memorised
    kwargs = {"eos_token_id": 0}
    trained = evaluate_model(model, features, REFERENCES, NAMES, tokenizer, GREEDY, **kwargs)
    untrained = evaluate_model(
        make_tiny_model(seed=1), features, REFERENCES, NAMES, tokenizer, GREEDY, **kwargs
    )
    assert untrained.scores.cider < trained.scores.cider
    assert untrained.scores.bleu1 < trained.scores.bleu1


def test_captions_do_not_depend_on_batch_size(memorised: Setup) -> None:
    _, features, tokenizer = memorised
    model = make_tiny_model(seed=2)
    generation = GenerationConfig(strategy="beam", max_new_tokens=6, beam_size=2)
    results = [
        generate_captions(
            model, features, NAMES, tokenizer, generation, eos_token_id=0, batch_size=batch_size
        )
        for batch_size in (1, 2, 64)
    ]
    assert results[0] == results[1] == results[2]


def test_sampling_is_reproducible_through_the_seed(memorised: Setup) -> None:
    _, features, tokenizer = memorised
    model = make_tiny_model(seed=2)
    generation = GenerationConfig(strategy="nucleus", max_new_tokens=8, top_p=1.0)

    def sample(seed: int) -> dict[str, str]:
        return generate_captions(
            model, features, NAMES, tokenizer, generation, eos_token_id=0, seed=seed
        )

    assert sample(3) == sample(3)
    assert sample(3) != sample(4)


def test_saved_json_follows_the_documented_schema(memorised: Setup, tmp_path: Path) -> None:
    model, features, tokenizer = memorised
    result = evaluate_model(model, features, REFERENCES, NAMES, tokenizer, GREEDY, eos_token_id=0)
    path = tmp_path / "nested" / "test.json"
    result.save(path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == result.to_dict()
    assert set(saved) == {"split", "num_images", "scores", "generation", "captions"}
    assert saved["split"] == "test"
    assert saved["num_images"] == 5
    assert set(saved["scores"]) == {"bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "cider"}
    assert saved["generation"]["strategy"] == "greedy"
    assert saved["captions"][0] == {
        "image": "dog.jpg",
        "caption": "a dog runs in the park",
        "references": ["a dog runs in the park"],
    }
    assert [item["image"] for item in saved["captions"]] == list(NAMES)


def test_model_mode_is_preserved(memorised: Setup) -> None:
    model, features, tokenizer = memorised
    assert not model.training
    generate_captions(model, features, NAMES[:1], tokenizer, GREEDY, eos_token_id=0)
    assert not model.training


@pytest.mark.parametrize(
    ("names", "kwargs", "error", "message"),
    [
        (("ghost.jpg",), {}, KeyError, "no cached embedding for 1 image"),
        ((), {}, ValueError, "no image to caption"),
        (NAMES, {"batch_size": 0}, ValueError, "batch_size must be positive"),
    ],
)
def test_invalid_generation_requests_are_rejected(
    memorised: Setup,
    names: tuple[str, ...],
    kwargs: dict[str, int],
    error: type[Exception],
    message: str,
) -> None:
    model, features, tokenizer = memorised
    with pytest.raises(error, match=message):
        generate_captions(model, features, names, tokenizer, GREEDY, eos_token_id=0, **kwargs)


def test_images_without_reference_are_rejected_before_generating(memorised: Setup) -> None:
    model, features, tokenizer = memorised
    references = {**REFERENCES, "cat.jpg": []}
    with pytest.raises(KeyError, match="no reference caption for 1 image"):
        evaluate_model(model, features, references, NAMES, tokenizer, GREEDY, eos_token_id=0)
