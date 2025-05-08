# test_mlp_adapter.py
#
# Unit tests for friday.model.vision_adapter.MLPAdapter.
#
# Run with:
#     pytest -q test_mlp_adapter.py
#
import tempfile
import torch
import pytest

# Robust import  -------------------------------------------------------------- #
try:
    from friday.model.vision_adapter import MLPAdapter
except ModuleNotFoundError:
    # fall‑back if project layout differs
    from friday.model import MLPAdapter


# --------------------------------------------------------------------------- #
# 1. Forward‑pass shape & dtype
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch, tokens, in_dim, hid_dim, out_dim, layers", [
    (2, 5, 32, 16, 64, 3),
    (1, 4, 128, 256, 32, 1),
])
def test_mlp_forward_dimension(batch, tokens, in_dim, hid_dim, out_dim, layers):
    torch.manual_seed(0)
    adapter = MLPAdapter(
        input_dim=in_dim,
        hidden_dim=hid_dim,
        output_dim=out_dim,
        num_layers=layers,
        activation="gelu",
    )
    x = torch.randn(batch, tokens, in_dim)
    y = adapter(x)

    assert y.shape == (batch, tokens, out_dim)
    assert y.dtype == x.dtype


# --------------------------------------------------------------------------- #
# 2. Activation switch logic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("act_cls, act_name", [
    (torch.nn.GELU, "gelu"),
    (torch.nn.ReLU, "relu"),
])
def test_activation_switch(act_cls, act_name):
    adapter = MLPAdapter(
        input_dim=8,
        hidden_dim=8,
        output_dim=8,
        num_layers=2,
        activation=act_name,
    )
    # Activation module should appear at least once in .mlp children
    activations = [m for m in adapter.mlp.modules() if isinstance(m, act_cls)]
    assert len(activations) >= 1


def test_invalid_activation_raises():
    with pytest.raises(ValueError):
        MLPAdapter(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            num_layers=2,
            activation="swish",
        )


# --------------------------------------------------------------------------- #
# 3. Checkpoint loading round‑trip
# --------------------------------------------------------------------------- #
def test_checkpoint_loading_roundtrip(tmp_path):
    """Save adapter weights, reload via checkpoint_path and ensure equality."""
    adapter1 = MLPAdapter(
        input_dim=16,
        hidden_dim=8,
        output_dim=4,
        num_layers=2,
        activation="relu",
    )
    ckpt_path = tmp_path / "adapter.pt"
    torch.save(adapter1.state_dict(), ckpt_path)

    adapter2 = MLPAdapter(
        input_dim=16,
        hidden_dim=8,
        output_dim=4,
        num_layers=2,
        activation="relu",
        checkpoint_path=str(ckpt_path),
    )

    # Parameter tensors must match exactly
    for p1, p2 in zip(adapter1.parameters(), adapter2.parameters()):
        assert torch.allclose(p1, p2)
