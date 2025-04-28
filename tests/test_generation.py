import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from PIL import Image

from friday.model import FridayForCausalLM
 
torch.random.manual_seed(0)

USE_FRIDAY=True

if USE_FRIDAY:
    model = FridayForCausalLM.from_pretrained(
        "microsoft/Phi-4-mini-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained('kevin510/friday')
    model.get_model().initialize_vision_modules()
else:
    model_path = "microsoft/Phi-4-mini-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)


image_1 = Image.open("tests/assets/cat_1.jpeg").convert("RGB")
image_2 = Image.open("tests/assets/cat_2.jpg").convert("RGB")
generation_args = {
    "max_new_tokens": 100,
    "return_full_text": False,
    "do_sample": False,
}
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def run_batch(examples):
    prompts = [ex["messages"] for ex in examples]
    imgs    = [ex.get("images", None) for ex in examples]

    outputs = pipe(prompts, images=imgs, batch_size=len(prompts), **generation_args)

    results = [o[0]["generated_text"] for o in outputs]
    print(f"Generated {len(results)} outputs.")
    print(results)





# PASSES
run_batch([
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Are you conscious?"},
        ],
        "images": []
    }
])


# PASSES
run_batch([
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is in this image: <image>"},
        ],
        "images": [image_1]
    }
])

# PASSES (Empty Response)
run_batch([
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the difference between this image: <image> and this image: <image>?"},
        ],
        "images": [image_1, image_2]
    }
])

# PASSES
run_batch([
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the difference between these two images?<image><image>"},
        ],
        "images": [image_1, image_2]
    }
])

# # PASSES
run_batch([
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Describe this image"},
        ],
        "images": [image_1]
    }
])

# PASSES
run_batch([
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Describe this image: <image>"},
        ],
        "images": []
    }
])

# PASSES
run_batch([
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
    },
])

# PASSES
run_batch([
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


