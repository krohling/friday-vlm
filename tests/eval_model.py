import json
import transformers
import torch, pytest, random
from torch.nn.functional import cross_entropy
from PIL import Image
from friday.train.config import FridayTrainingArguments
from friday.data import PretrainingDataset, FinetuningDataset, FridayCollator
from friday.train.model_factory import build_model

from peft import PeftModel, PeftConfig

CHECKPOINT_PATH = "checkpoints-finetune/friday/final_checkpoint"
VISION_ADAPTER_PATH = f"{CHECKPOINT_PATH}/mm_projector.bin"
LORA_ADAPTER_PATH = f"{CHECKPOINT_PATH}/lora"

example_count = 2

with open("./config/finetune.json", 'r') as f:
    config = json.load(f)

config['training']['lora_enable'] = False
config['model']['cfg_vision_adapter']['checkpoint_path'] = VISION_ADAPTER_PATH
model, tokenizer = build_model(
    config['model'],
    config['tokenizer'],
    FridayTrainingArguments(**config['training']),
)
model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH, is_trainable=False)
model.print_device_configuration()
model.to(device=model.device, dtype=torch.bfloat16)
model.eval()

collator = FridayCollator(tokenizer=tokenizer)
dataset = FinetuningDataset(
    data_path="datasets/llava_v1_5_mix665k/llava_v1_5_mix665k_filtered.json",
    image_dir="datasets/llava_v1_5_mix665k/",
    tokenizer=tokenizer,
    vision_tower=model.get_vision_tower(),
)

for step in range(example_count):
    sample = dataset[random.randint(0, len(dataset)-1)]
    batch = collator([sample])

    batch['input_ids'] = batch['input_ids'].to(device=model.device)
    batch['labels'] = batch['labels'].to(device=model.device)
    batch['attention_mask'] = batch['attention_mask'].to(device=model.device)
    for i in range(len(batch['images'])):
        for j in range(len(batch['images'][i])):
            batch['images'][i][j] = batch['images'][i][j].to(device=model.device)

    # ───────── forward pass ─────────
    with torch.no_grad():
        outputs = model(**batch)
    
    # detokenize and print the output as well as the ground truth
    output_text = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
    
    labels_for_decode = batch['labels'].detach().clone()
    pad_id = tokenizer.pad_token_id or 0
    labels_for_decode[labels_for_decode == -100] = pad_id
    target_text = tokenizer.batch_decode(
        labels_for_decode.cpu().tolist(),
        skip_special_tokens=True
    )

    print(f"Step {step+1}/{example_count} - Output: {output_text}, Target: {target_text}")

