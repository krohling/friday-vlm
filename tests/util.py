import pytest
from pathlib import Path
from PIL import Image

def build_test_images_paths():
    current_file_path = Path(__file__).resolve()
    return [
        current_file_path.parent / "assets" / "images" / "cat_1.jpeg", 
        current_file_path.parent / "assets" / "images" / "cat_2.jpeg"
    ]

@pytest.fixture()
def test_images_paths():
    return build_test_images_paths()

@pytest.fixture()
def test_images(test_images_paths):
    return [Image.open(p).convert("RGB") for p in test_images_paths]