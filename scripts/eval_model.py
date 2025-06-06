import json
import transformers
import torch, pytest, random
from torch.nn.functional import cross_entropy
from PIL import Image
from friday.train.config import FridayTrainingArguments
from friday.data import PretrainingDataset, FinetuningDataset, FridayCollator
from friday.train.model_factory import build_model

from peft import PeftModel, PeftConfig



CHECKPOINT_PATH = "./checkpoint-4000"
VISION_ADAPTER_PATH = f"{CHECKPOINT_PATH}/mm_projector.bin"
LORA_ADAPTER_PATH = f"{CHECKPOINT_PATH}/lora"

example_count = 2

with open("./config/finetune.json", 'r') as f:
    config = json.load(f)

config['model']["attn_implementation"] = None
config['training']['lora_enable'] = False
config['model']['cfg_vision_adapter']['checkpoint_path'] = VISION_ADAPTER_PATH
print(config)
model, tokenizer = build_model(
    config['model'],
    config['tokenizer'],
    FridayTrainingArguments(**config['training']),
)
model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH, is_trainable=False)
model.print_device_configuration()
model.to(device=model.device, dtype=torch.bfloat16)
model.eval()


# image_path = "datasets/llava_v1_5_mix665k_small/coco/train2017/000000002963.jpg"
image_path = "freshpoint-produce-101-banana-v2.jpg"
prompt = "Please describe this image."
max_tokens = 128
device = model.device

# ───────────────────── preprocess the image ─────────────────
image = Image.open(image_path).convert("RGB")

vision_tower = model.get_vision_tower()                     # FastViT-HD
pixel = vision_tower.image_processor(                       # same transforms as training
    image, return_tensors="pt"
)["pixel_values"].to(device, dtype=torch.bfloat16)           # (1, 3, H, W)
images = [[pixel[0]]]                                        # shape expected by model

# ───────────────────── build text prompt ────────────────────
user_prompt = f"<|user|><image>\n{prompt}\n<|assistant|>"
tok = tokenizer(user_prompt, return_tensors="pt")
input_ids      = tok["input_ids"].to(device)
attention_mask = tok["attention_mask"].to(device)

eos_id = tokenizer.convert_tokens_to_ids("<|end|>")

# ───────────────────────── generate ─────────────────────────
with torch.no_grad():
    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        max_new_tokens=max_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
    )

answer = tokenizer.decode(gen_ids[0], skip_special_tokens=False).strip()
answer = answer.split("<|assistant|>")[1].replace("<|end|>", "").strip()  # remove assistant token and EOS
print("\n🖼️  Image:", image_path)
print("📋 Prompt:", prompt)
print("🗣️  Model :", answer)