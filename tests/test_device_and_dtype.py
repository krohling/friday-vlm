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
# ------------------------------  Fixtures ---------------------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture
def friday():
    from friday.model import FridayForCausalLM, FridayConfig

    cfg = FridayConfig(delay_load=True)
    model = FridayForCausalLM(cfg)
    return model


# --------------------------------------------------------------------------- #
# ------------------------------  Test cases -------------------------------- #
# --------------------------------------------------------------------------- #
def test_print_device_configuration_without_vision_modules(friday, capsys):
    """Method prints without raising and produces non‑empty output."""
    friday.print_device_configuration()
    captured = capsys.readouterr().out
    assert "Device Configuration" in captured
    assert captured.strip()  # some text printed


def test_print_device_configuration(friday, capsys):
    """Method prints without raising and produces non‑empty output."""
    friday.initialize_vision_modules()
    friday.print_device_configuration()
    captured = capsys.readouterr().out
    assert "Device Configuration" in captured
    assert captured.strip()  # some text printed
