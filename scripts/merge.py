import json
import torch
from torch.nn.functional import cross_entropy
from friday.train.config import FridayTrainingArguments
from friday.train.model_factory import build_model

from peft import PeftModel, PeftConfig



CHECKPOINT_PATH = "./model"
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

merged = model.merge_and_unload()
merged.save_pretrained("./model-bf16", safe_serialization=True)
