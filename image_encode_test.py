import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from easydict import EasyDict
from PIL import Image

from model import build_friday_phi
 
torch.random.manual_seed(0)

USE_FRIDAY=True

if USE_FRIDAY:
    config = EasyDict({
        "vision_tower": {
            "vision_tower": "google/siglip2-base-patch16-384",
            # "vision_tower": "google/siglip2-so400m-patch16-384",
            "s2_scales": "384,768",
        },
        "vision_adapter": {
            "input_dim": 1536,
            "hidden_dim": 512,
            "output_dim": 3072,
            "layers": 2,
            "activation": "gelu"
        },
    })
    model, tokenizer = build_friday_phi(config)
else:
    model_path = "microsoft/Phi-4-mini-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

image_1 = Image.open("cat_1.jpeg").convert("RGB")
image_2 = Image.open("cat_2.jpg").convert("RGB")
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

def run_batch(examples, **gen_kw):
    prompts = [ex["messages"] for ex in examples]
    imgs    = [ex.get("images", None) for ex in examples]

    outputs = pipe(prompts, images=imgs, batch_size=len(prompts), **gen_kw)

    results = [o[0]["generated_text"] for o in outputs]
    print(f"Generated {len(results)} outputs.")
    print(results)





# PASSES
# run_batch([
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": "Are you conscious?"},
#         ],
#         "images": []
#     }
# ])


# run_batch([
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": "Describe this image: <image>"},
#         ],
#         "images": [image_1]
#     }
# ])

# # OOM
# run_batch([
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": "What is the difference between this image: <image> and this image: <image>?"},
#         ],
#         "images": [image_1, image_2]
#     }
# ])

# OOM
# run_batch([
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": "What is the difference between these two images?<image><image>"},
#         ],
#         "images": [image_1, image_2]
#     }
# ])

# PASSES
# run_batch([
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": "Describe this image"},
#         ],
#         "images": [image_1]
#     }
# ])

# PASSES
# run_batch([
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": "Describe this image: <image>"},
#         ],
#         "images": []
#     }
# ])

# PASSES
# run_batch([
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": "Are you conscious?"},
#         ],
#         "images": []
#     },
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": "Describe the sky in a poetic prose please."},
#         ],
#         "images": []
#     },
# ])

# FAILS
# run_batch([
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": "Are you conscious?"},
#         ],
#         "images": []
#     },
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": "Describe this image: <image>"},
#         ],
#         "images": [image_1]
#     },
#     # {
#     #     "messages": [
#     #         {"role": "system", "content": "You are a helpful AI assistant."},
#     #         {"role": "user", "content": "What is the difference between this image: <image> and this image: <image>?"},
#     #     ],
#     #     "images": [image_1, image_2]
#     # }
# ])


