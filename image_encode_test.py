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
            "output_dim": 256,
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


messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Describe this image.<image>"},
]


image = Image.open("cat.jpeg").convert("RGB")
 
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    images=[image],
)
 
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
 
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])