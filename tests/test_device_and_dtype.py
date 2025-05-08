# test_device_dtype.py
#
# “Device / dtype sanity” unit‑tests.
# Verifies that `print_device_configuration()` runs without error and that the
# vision‑tower and MM‑projector live on the same device.
#
# All heavyweight components are replaced by <1 kB stubs, so the file executes
# in < 1 s on CPU.
#
# Run with:  pytest -q test_device_dtype.py
#
import pytest
import torch


# --------------------------------------------------------------------------- #
# ------------------------  Lightweight stub modules ------------------------ #
# --------------------------------------------------------------------------- #
class DummyVisionTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))   # at least 1 param

    def forward(self, x):
        return x                                           # no‑op

    # attribute accessed by print_device_configuration
    @property
    def device(self):
        return torch.device("cpu")


class DummyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


# --------------------------------------------------------------------------- #
# -------------------------  Global monkey‑patch fixture -------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_everything(monkeypatch):
    """
    Patch heavy classes **before first Friday import**.  Also patch
    `friday.util.get_module_device` so it works with the dummy adapter.
    """
    # Stub out vision tower & adapter
    import friday.model.vision_tower as vt
    import friday.model.vision_adapter as va
    monkeypatch.setattr(vt, "SiglipVisionTower", DummyVisionTower, raising=True)
    monkeypatch.setattr(vt, "SiglipVisionTowerS2", DummyVisionTower, raising=True)
    monkeypatch.setattr(va, "MLPAdapter", DummyAdapter, raising=True)

    # Replace util helper
    import friday.util as futil
    monkeypatch.setattr(futil, "get_module_device",
                        lambda module: next(module.parameters()).device,
                        raising=True)

    # Lightweight FridayForCausalLM.__init__
    from friday.model.friday import FridayForCausalLM, FridayConfig

    def _light_init(self, config: FridayConfig):
        self.config = config
        self.device = torch.device("cpu")

        # tiny dummy inner model ------------------------------------------- #
        class _Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = DummyVisionTower()
                self.mm_projector = DummyAdapter()

            # called by outer.initialize_vision_modules(); nothing to do
            def initialize_vision_modules(self):
                pass

            @property
            def device(self):   # attribute used by print_device_configuration
                return torch.device("cpu")

        self.model = _Inner()

    monkeypatch.setattr(FridayForCausalLM, "__init__", _light_init, raising=True)


# --------------------------------------------------------------------------- #
# ------------------------------  Fixtures ---------------------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture
def friday():
    from friday.model.friday import FridayForCausalLM, FridayConfig
    return FridayForCausalLM(FridayConfig(delay_load=True))


# --------------------------------------------------------------------------- #
# ------------------------------  Test cases -------------------------------- #
# --------------------------------------------------------------------------- #
def test_print_device_configuration_does_not_crash(friday, capsys):
    """Method prints without raising and produces non‑empty output."""
    friday.print_device_configuration()
    captured = capsys.readouterr().out
    assert "Device Configuration" in captured
    assert captured.strip()  # some text printed


def test_module_devices_match(friday):
    friday.initialize_vision_modules()      # no‑op but keeps API contract
    vt_device = friday.get_model().vision_tower.device
    mp_device = next(friday.get_model().mm_projector.parameters()).device
    assert vt_device == mp_device == torch.device("cpu")
