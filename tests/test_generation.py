import json
from easydict import EasyDict
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from PIL import Image
from friday.model import FridayForCausalLM

USE_FRIDAY = True

# load model.json config
with open("tests/assets/model.json", "r") as f:
    model_config = EasyDict(json.load(f))


@pytest.fixture(scope="module")
def model_and_tokenizer():
    torch.random.manual_seed(0)

    if USE_FRIDAY:
        model = FridayForCausalLM.from_pretrained(
            model_config.language_model.model_name_or_path,
            cfg_vision_tower=model_config.vision_tower,
            cfg_vision_adapter=model_config.vision_adapter,
            **model_config.language_model.model_params,
            torch_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.language_model.tokenizer_name_or_path,
            **model_config.language_model.tokenizer_params,
        )

        model.get_model().initialize_vision_modules()
        model.print_device_configuration()
    else:
        model_path = "microsoft/Phi-4-mini-instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return model, tokenizer, pipe

@pytest.fixture(scope="module")
def images():
    image_1 = Image.open("tests/assets/cat_1.jpeg").convert("RGB")
    image_2 = Image.open("tests/assets/cat_2.jpg").convert("RGB")
    return image_1, image_2

GEN_ARGS = {
    "max_new_tokens": 100,
    "return_full_text": False,
    "do_sample": False,
}

def run_batch(pipe, examples, **kwargs):
    prompts = [ex["messages"] for ex in examples]
    imgs    = [ex.get("images", None) for ex in examples]
    outputs = pipe(prompts, images=imgs, batch_size=len(prompts), **GEN_ARGS)
    return [o[0]["generated_text"] for o in outputs]

# === TESTS ===

def test_basic_text(model_and_tokenizer):
    _, _, pipe = model_and_tokenizer
    results = run_batch(pipe, [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Are you conscious?"},
            ],
            "images": []
        }
    ])
    assert isinstance(results[0], str)

def test_single_image(model_and_tokenizer, images):
    model, _, pipe = model_and_tokenizer
    image_1, _ = images
    results = run_batch(pipe, [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is in this image: <image>"},
            ],
            "images": [image_1]
        }
    ])
    assert isinstance(results[0], str)

def test_single_image_preprocessed(model_and_tokenizer, images):
    model, _, pipe = model_and_tokenizer
    image_1, _ = images
    results = run_batch(pipe, [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is in this image: <image>"},
            ],
            "images": model.get_model().vision_tower.preprocess_images([image_1])
        }
    ])
    assert isinstance(results[0], str)

def test_two_images(model_and_tokenizer, images):
    model, _, pipe = model_and_tokenizer
    image_1, image_2 = images
    results = run_batch(pipe, [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the difference between this image: <image> and this image: <image>?"},
            ],
            "images": [image_1, image_2]
        }
    ])
    assert isinstance(results[0], str)

def test_two_images_preprocessed(model_and_tokenizer, images):
    model, _, pipe = model_and_tokenizer
    image_1, image_2 = images
    results = run_batch(pipe, [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the difference between this image: <image> and this image: <image>?"},
            ],
            "images": model.get_model().vision_tower.preprocess_images([image_1, image_2])
        }
    ])
    assert isinstance(results[0], str)


def test_missing_image_1(model_and_tokenizer):
    _, _, pipe = model_and_tokenizer
    results = run_batch(pipe, [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Describe this image: <image>"},
            ],
            "images": []
        }
    ])
    assert isinstance(results[0], str)

def test_missing_image_2(model_and_tokenizer, images):
    model, _, pipe = model_and_tokenizer
    image_1, _ = images
    with pytest.raises(ValueError):
        run_batch(pipe, [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is the difference between this image: <image> and this image: <image>?"},
                ],
                "images": [image_1]
            }
        ])


def test_batch_no_images(model_and_tokenizer):
    _, _, pipe = model_and_tokenizer
    results = run_batch(pipe, [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Are you conscious?"},
            ],
            "images": []
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "I would like for you to describe the sky in a poetic prose please."},
            ],
            "images": []
        }
    ])
    assert all(isinstance(r, str) for r in results)


def test_batch_text_and_images(model_and_tokenizer, images):
    _, _, pipe = model_and_tokenizer
    image_1, image_2 = images
    results = run_batch(pipe, [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Are you conscious?"},
            ],
            "images": []
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is in this image: <image>"},
            ],
            "images": [image_1]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the difference between this image: <image> and this image: <image>?"},
            ],
            "images": [image_1, image_2]
        }
    ])
    assert all(isinstance(r, str) for r in results)


def test_batch_preprocessed_images(model_and_tokenizer, images):
    model, _, pipe = model_and_tokenizer
    image_1, image_2 = images
    results = run_batch(pipe, [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Are you conscious?"},
            ],
            "images": []
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is in this image: <image>"},
            ],
            "images": model.get_model().vision_tower.preprocess_images([image_1])
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the difference between this image: <image> and this image: <image>?"},
            ],
            "images": model.get_model().vision_tower.preprocess_images([image_1, image_2])
        }
    ])
    assert all(isinstance(r, str) for r in results)

