import pytest
import torch
from torch.utils.data import DataLoader

from clipcap.data.dataset import CaptionBatch, CaptionDataset, collate_captions
from tests.helpers import WordTokenizer, make_features

NAMES = ("a.jpg", "b.jpg", "c.jpg")
CAPTIONS = {
    "a.jpg": ["a dog runs", "a brown dog runs fast in the park"],
    "b.jpg": ["a cat"],
    "c.jpg": ["two children play"],
}
EOS = WordTokenizer.eos_token_id


def _dataset(names: tuple[str, ...] = NAMES, max_tokens: int = 40) -> CaptionDataset:
    return CaptionDataset(
        make_features(NAMES),
        CAPTIONS,
        names,
        WordTokenizer(),
        eos_token_id=EOS,
        max_tokens=max_tokens,
    )


def test_one_example_per_caption() -> None:
    dataset = _dataset()
    assert len(dataset) == 4
    assert [dataset.image_name(index) for index in range(4)] == ["a.jpg", "a.jpg", "b.jpg", "c.jpg"]
    assert dataset.embedding_dim == 8


def test_examples_pair_each_caption_with_its_image_embedding() -> None:
    features = make_features(NAMES)
    tokenizer = WordTokenizer()
    dataset = CaptionDataset(features, CAPTIONS, ("c.jpg", "a.jpg"), tokenizer, eos_token_id=EOS)
    embedding, tokens = dataset[0]
    assert torch.equal(embedding, features.embeddings[2])
    assert tokenizer.decode(tokens.tolist()) == "two children play <eos>"
    assert torch.equal(dataset[1][0], features.embeddings[0])
    assert torch.equal(dataset[2][0], features.embeddings[0])


def test_every_caption_ends_with_eos_even_when_truncated() -> None:
    dataset = _dataset(max_tokens=4)
    for index in range(len(dataset)):
        tokens = dataset[index][1]
        assert tokens.dtype == torch.long
        assert tokens[-1] == EOS
        assert len(tokens) <= 4
        assert (tokens[:-1] != EOS).all()
    assert len(dataset[1][1]) == 4  # the long caption was truncated


def test_only_requested_images_are_used() -> None:
    dataset = _dataset(names=("b.jpg",))
    assert len(dataset) == 1
    assert dataset.image_name(0) == "b.jpg"


def test_collate_pads_and_masks() -> None:
    dataset = _dataset()
    batch = dataset.collate([dataset[0], dataset[1], dataset[2]])
    assert isinstance(batch, CaptionBatch)
    assert len(batch) == 3
    assert batch.embeddings.shape == (3, 8)
    assert batch.tokens.shape == batch.mask.shape == (3, 9)
    assert batch.mask.dtype == torch.bool
    assert batch.mask.sum(dim=1).tolist() == [4, 9, 3]
    for row in range(3):
        length = int(batch.mask[row].sum())
        assert torch.equal(batch.tokens[row, :length], dataset[row][1])
        assert batch.mask[row, :length].all()
        assert not batch.mask[row, length:].any()
        assert (batch.tokens[row, length:] == EOS).all()


def test_collate_uses_the_given_padding_id() -> None:
    dataset = _dataset()
    batch = collate_captions([dataset[0], dataset[1]], pad_token_id=99)
    assert (batch.tokens[0, 4:] == 99).all()


def test_batch_can_be_moved_to_a_device() -> None:
    dataset = _dataset()
    moved = dataset.collate([dataset[0]]).to("cpu")
    assert moved.tokens.device.type == "cpu"
    assert moved.embeddings.shape == (1, 8)


def test_works_with_a_dataloader() -> None:
    dataset = _dataset()
    loader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate)
    batches = list(loader)
    assert [len(batch) for batch in batches] == [3, 1]
    assert sum(int(batch.mask.sum()) for batch in batches) == sum(
        len(dataset[index][1]) for index in range(len(dataset))
    )


@pytest.mark.parametrize(
    ("names", "captions", "max_tokens", "error", "message"),
    [
        (("z.jpg",), CAPTIONS, 40, KeyError, "no cached embedding"),
        (("a.jpg",), {"a.jpg": []}, 40, KeyError, "no caption"),
        (("a.jpg",), {}, 40, KeyError, "no caption"),
        (("a.jpg",), {"a.jpg": ["   "]}, 40, ValueError, "has no token"),
        (("a.jpg",), CAPTIONS, 1, ValueError, "at least 2"),
        ((), CAPTIONS, 40, ValueError, "dataset is empty"),
    ],
)
def test_invalid_inputs_are_rejected(
    names: tuple[str, ...],
    captions: dict[str, list[str]],
    max_tokens: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        CaptionDataset(
            make_features(NAMES),
            captions,
            names,
            WordTokenizer(),
            eos_token_id=EOS,
            max_tokens=max_tokens,
        )


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_captions([], pad_token_id=EOS)
