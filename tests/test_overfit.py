import transformers
import torch, pytest, random
from torch.nn.functional import cross_entropy
from PIL import Image
from friday.data import PretrainingDataset, FinetuningDataset, FridayCollator


STEPS   = 100
TARGET  = 0.05

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize(
    "dataset_type",
    [
        "pretrain",
        "finetuning",
    ],
)
def test_tiny_overfit(dataset_type):
    from friday.model import FridayForCausalLM, FridayConfig
    cfg = FridayConfig()

    model = FridayForCausalLM(cfg).to(dtype=torch.bfloat16, device='cuda')
    model.initialize_vision_modules()
    model.cuda()
    model.set_vision_tower_dtype(torch.bfloat16)

    tokenizer = transformers.AutoTokenizer.from_pretrained('kevin510/friday')

    # ───────── build one synthetic batch (you can load a real datum instead) ────────
    if dataset_type == "finetuning":
        print("Testing overfit on finetuning dataset")
        model.set_llm_requires_grad(True)
        model.set_vision_tower_requires_grad(False)
        model.set_vision_adapter_requires_grad(False)
        dataset = FinetuningDataset(
            data_path="./datasets/llava_v1_5_mix665k_small/llava_v1_5_mix665k_small.json",
            image_dir="./datasets/llava_v1_5_mix665k_small/",
            tokenizer=tokenizer,
            vision_tower=model.get_vision_tower(),
            max_count=2
        )
    else:
        print("Testing overfit on pretraining dataset")
        model.set_llm_requires_grad(False)
        model.set_vision_tower_requires_grad(False)
        model.set_vision_adapter_requires_grad(True)
        dataset = PretrainingDataset(
            data_path="./datasets/LLaVA-Pretrain_small/blip_laion_cc_sbu_558k_meta_small.json",
            image_dir="./datasets/LLaVA-Pretrain_small/images",
            tokenizer=tokenizer,
            vision_tower=model.get_vision_tower(),
            max_count=2
        )
    
    optim = torch.optim.AdamW(model.model.mm_projector.parameters(), lr=0.0003)
    model.print_device_configuration()
    collator = FridayCollator(
        tokenizer=tokenizer
    )

    for step in range(STEPS):
        sample = dataset[random.randint(0, len(dataset)-1)]
        batch = collator([sample])

        batch['input_ids'] = batch['input_ids'].cuda()
        batch['labels'] = batch['labels'].cuda()
        batch['attention_mask'] = batch['attention_mask'].cuda()
        for i in range(len(batch['images'])):
            for j in range(len(batch['images'][i])):
                batch['images'][i][j] = batch['images'][i][j].cuda()

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
