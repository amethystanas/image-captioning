import pytest
import torch
from torch import nn

from clipcap.models.mapper import MLPMapper

CLIP_DIM, GPT_DIM, PREFIX = 12, 16, 5


def _mlp() -> nn.Module:
    return MLPMapper(CLIP_DIM, GPT_DIM, PREFIX)


MAPPERS = {"mlp": _mlp}


@pytest.fixture(params=sorted(MAPPERS))
def mapper(request: pytest.FixtureRequest) -> nn.Module:
    torch.manual_seed(0)
    return MAPPERS[request.param]().eval()


@pytest.mark.parametrize("batch_size", [1, 4])
def test_output_shape(mapper: nn.Module, batch_size: int) -> None:
    prefix = mapper(torch.randn(batch_size, CLIP_DIM))
    assert prefix.shape == (batch_size, PREFIX, GPT_DIM)
    assert torch.isfinite(prefix).all()


def test_every_parameter_receives_a_gradient(mapper: nn.Module) -> None:
    mapper(torch.randn(3, CLIP_DIM)).pow(2).sum().backward()
    for name, parameter in mapper.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name


def test_each_example_is_mapped_independently(mapper: nn.Module) -> None:
    embeddings = torch.randn(4, CLIP_DIM)
    together = mapper(embeddings)
    alone = torch.cat([mapper(embeddings[row : row + 1]) for row in range(4)])
    assert torch.allclose(together, alone, atol=1e-5)


def test_different_images_give_different_prefixes(mapper: nn.Module) -> None:
    prefix = mapper(torch.randn(2, CLIP_DIM))
    assert not torch.allclose(prefix[0], prefix[1])


@pytest.mark.parametrize("shape", [(CLIP_DIM,), (2, CLIP_DIM + 1), (2, 3, CLIP_DIM)])
def test_wrong_input_shape_is_rejected(mapper: nn.Module, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="expected embeddings of shape"):
        mapper(torch.randn(*shape))


def test_mlp_hidden_size_follows_ratio() -> None:
    mapper = MLPMapper(CLIP_DIM, GPT_DIM, PREFIX, hidden_ratio=0.25)
    assert mapper.network[0].out_features == PREFIX * GPT_DIM // 4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"clip_dim": 0}, "clip_dim must be positive"),
        ({"gpt_dim": -1}, "gpt_dim must be positive"),
        ({"prefix_length": 0}, "prefix_length must be positive"),
        ({"hidden_ratio": 0.0}, "hidden_ratio must be positive"),
    ],
)
def test_mlp_rejects_invalid_sizes(kwargs: dict[str, float], message: str) -> None:
    arguments = {"clip_dim": CLIP_DIM, "gpt_dim": GPT_DIM, "prefix_length": PREFIX, **kwargs}
    with pytest.raises(ValueError, match=message):
        MLPMapper(**arguments)  # type: ignore[arg-type]
