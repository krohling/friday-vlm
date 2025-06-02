import json
import transformers
import torch, pytest, random
from torch.nn.functional import cross_entropy
from PIL import Image
from friday.train.config import FridayTrainingArguments
from friday.data import PretrainingDataset, FinetuningDataset, FridayCollator
from friday.train.model_factory import build_model


STEPS   = 300
TARGET  = 0.1

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize(
    "config_path, data_path, image_dir",
    [
        (
            "./config/pretrain.json",
            "datasets/LLaVA-Pretrain_small/blip_laion_cc_sbu_558k_meta_small.json",
            "datasets/LLaVA-Pretrain_small/images"
        ),
        (
            "./config/finetune.json",
            "datasets/llava_v1_5_mix665k_small/llava_v1_5_mix665k_small.json",
            "datasets/llava_v1_5_mix665k_small/",
        )
    ],
)
def test_tiny_overfit(config_path, data_path, image_dir):
    print("Testing overfit with config:", config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    del config['model']['attn_implementation']
    model, tokenizer = build_model(
        config['model'],
        config['tokenizer'],
        FridayTrainingArguments(**config['training']),
    )
    collator = FridayCollator(tokenizer=tokenizer)
    model.print_device_configuration()
    model.to(device=model.device, dtype=torch.bfloat16)
    model.train()

    if "finetune" in config_path:
        dataset = FinetuningDataset(
            data_path=data_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            vision_tower=model.get_vision_tower(),
            max_count=2
        )
    else:
        dataset = PretrainingDataset(
            data_path=data_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            vision_tower=model.get_vision_tower(),
            max_count=2
        )
    

    # ───────── build one synthetic batch (you can load a real datum instead) ────────
    optim = torch.optim.AdamW(model.parameters(), lr=0.0003)

    for step in range(STEPS):
        sample = dataset[random.randint(0, len(dataset)-1)]
        batch = collator([sample])

        batch['input_ids'] = batch['input_ids'].to(device=model.device)
        batch['labels'] = batch['labels'].to(device=model.device)
        batch['attention_mask'] = batch['attention_mask'].to(device=model.device)
        for i in range(len(batch['images'])):
            for j in range(len(batch['images'][i])):
                batch['images'][i][j] = batch['images'][i][j].to(device=model.device)

        # ───────── forward pass ─────────
        optim.zero_grad()
        outputs = model(**batch)

        # ───────── backward pass ─────────
        loss = outputs.loss
        loss.backward()
        optim.step()

        print(f"Step {step+1}/{STEPS} - Loss: {loss.item():.4f}")

        if loss.item() < TARGET:
            break


    assert loss.item() < TARGET, f"did not over‑fit (loss {loss.item():.3f})"
