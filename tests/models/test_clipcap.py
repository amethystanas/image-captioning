import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from clipcap.models.clipcap import ClipCaptionModel
from clipcap.models.mapper import MLPMapper
from tests.helpers import TINY_GPT_DIM, TINY_VOCAB, make_tiny_gpt2, make_tiny_model

CLIP_DIM, PREFIX = 8, 3


def _batch(seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """Three captions of lengths 5, 3 and 2, padded to 5."""
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(3, CLIP_DIM, generator=generator)
    tokens = torch.randint(1, TINY_VOCAB, (3, 5), generator=generator)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    return embeddings, tokens.masked_fill(~mask, 0), mask


@pytest.mark.parametrize("mapper_type", ["mlp", "transformer"])
def test_forward_returns_finite_scalar_loss_and_caption_logits(mapper_type: str) -> None:
    model = make_tiny_model(mapper_type=mapper_type)
    output = model(*_batch())
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (3, 5, TINY_VOCAB)


def test_untrained_loss_is_close_to_uniform_entropy() -> None:
    loss = make_tiny_model()(*_batch()).loss
    assert loss.item() == pytest.approx(torch.log(torch.tensor(float(TINY_VOCAB))).item(), rel=0.2)


def test_loss_is_the_mean_cross_entropy_over_real_tokens() -> None:
    model = make_tiny_model()
    embeddings, tokens, mask = _batch()
    output = model(embeddings, tokens, mask)
    per_token = F.cross_entropy(
        output.logits.reshape(-1, TINY_VOCAB), tokens.reshape(-1), reduction="none"
    ).view(3, 5)
    assert output.loss.item() == pytest.approx((per_token * mask).sum().item() / 10, rel=1e-5)


def test_padding_tokens_do_not_influence_the_loss() -> None:
    model = make_tiny_model()
    embeddings, tokens, mask = _batch()
    reference = model(embeddings, tokens, mask).loss
    scrambled = tokens.masked_fill(~mask, TINY_VOCAB - 1)
    assert model(embeddings, scrambled, mask).loss.item() == pytest.approx(
        reference.item(), rel=1e-5
    )


def test_padding_does_not_change_the_loss_of_a_caption() -> None:
    model = make_tiny_model()
    embeddings, tokens, mask = _batch()
    alone = model(embeddings[2:], tokens[2:, :2], mask[2:, :2]).loss
    padded = model(embeddings[2:], tokens[2:], mask[2:]).loss
    assert padded.item() == pytest.approx(alone.item(), rel=1e-5)


def test_first_caption_token_is_predicted_from_the_prefix_alone() -> None:
    model = make_tiny_model()
    embeddings, tokens, mask = _batch()
    other_tokens = tokens.clone()
    other_tokens[:, 0] = (tokens[:, 0] % (TINY_VOCAB - 1)) + 1
    first = model(embeddings, tokens, mask).logits
    second = model(embeddings, other_tokens, mask).logits
    assert torch.allclose(first[:, 0], second[:, 0], atol=1e-6)
    assert not torch.allclose(first[:, 1], second[:, 1], atol=1e-6)


def test_predictions_are_causal() -> None:
    model = make_tiny_model()
    embeddings, tokens, mask = _batch()
    changed = tokens.clone()
    changed[0, 3] = (tokens[0, 3] % (TINY_VOCAB - 1)) + 1
    first = model(embeddings, tokens, mask).logits[0]
    second = model(embeddings, changed, mask).logits[0]
    assert torch.allclose(first[:4], second[:4], atol=1e-6)
    assert not torch.allclose(first[4], second[4], atol=1e-6)


def test_predictions_depend_on_the_image() -> None:
    model = make_tiny_model()
    embeddings, tokens, mask = _batch()
    first = model(embeddings, tokens, mask).logits
    second = model(embeddings + 1.0, tokens, mask).logits
    assert not torch.allclose(first, second, atol=1e-4)


def test_frozen_gpt2_gets_no_gradient_and_is_not_updated() -> None:
    model = make_tiny_model(freeze_gpt2=True).train()
    before = {name: value.clone() for name, value in model.gpt2.state_dict().items()}
    trainable = model.trainable_parameters()
    assert {id(p) for p in trainable} == {id(p) for p in model.mapper.parameters()}

    optimizer = torch.optim.SGD(trainable, lr=0.1)
    model(*_batch()).loss.backward()
    optimizer.step()

    assert all(parameter.grad is None for parameter in model.gpt2.parameters())
    assert all(parameter.grad is not None for parameter in model.mapper.parameters())
    for name, value in model.gpt2.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_frozen_gpt2_stays_in_eval_mode_while_the_mapper_trains() -> None:
    model = make_tiny_model(freeze_gpt2=True).train()
    assert model.mapper.training
    assert not model.gpt2.training
    assert not any(module.training for module in model.gpt2.modules())


def test_unfrozen_gpt2_is_trained() -> None:
    model = make_tiny_model(freeze_gpt2=False).train()
    assert model.gpt2.training
    model(*_batch()).loss.backward()
    gradients = [parameter.grad for parameter in model.gpt2.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert len(model.trainable_parameters()) == len(list(model.parameters()))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e, t, m: (e, t, m[:, :4]), "must share"),
        (lambda e, t, m: (e, t[0], m[0]), "must share"),
        (lambda e, t, m: (e[:2], t, m), r"2 image embedding\(s\) for 3 caption"),
        (
            lambda e, t, m: (e, t.repeat(1, 6), m.repeat(1, 6)),
            "exceeds GPT-2's limit of 32",
        ),
    ],
)
def test_inconsistent_inputs_are_rejected(change, message: str) -> None:  # type: ignore[no-untyped-def]
    model = make_tiny_model()
    with pytest.raises(ValueError, match=message):
        model(*change(*_batch()))


def test_mapper_and_gpt2_sizes_must_agree() -> None:
    mapper = MLPMapper(CLIP_DIM, TINY_GPT_DIM + 4, PREFIX)
    with pytest.raises(ValueError, match="GPT-2 expects 16"):
        ClipCaptionModel(make_tiny_gpt2(), mapper)


@pytest.mark.slow
def test_from_config_loads_pretrained_gpt2() -> None:
    from clipcap.config import ModelConfig

    model = ClipCaptionModel.from_config(ModelConfig(prefix_length=4), clip_dim=512)
    assert model.max_positions == 1024
    tokens = torch.tensor([[64, 3290, 50256]])
    output = model(torch.randn(1, 512), tokens, torch.ones_like(tokens, dtype=torch.bool))
    assert output.logits.shape == (1, 3, 50257)
    assert not any(parameter.requires_grad for parameter in model.gpt2.parameters())
