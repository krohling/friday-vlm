from easydict import EasyDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from model.language_model.friday_phi import build_friday_phi
 
torch.random.manual_seed(0)

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
 
# from transformers import AutoTokenizer, Phi3ForCausalLM

# model = Phi3ForCausalLM.from_pretrained("meta-phi3/Phi3-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("meta-phi3/Phi3-2-7b-hf")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]