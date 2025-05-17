import transformers
import torch, pytest, random
from torch.nn.functional import cross_entropy
from PIL import Image
from friday.train.data import PretrainingDataset, PretrainingCollator

STEPS   = 60
TARGET  = 0.05

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_tiny_overfit():
    from friday.model import FridayForCausalLM, FridayConfig
    cfg = FridayConfig()

    model = FridayForCausalLM(cfg).to(dtype=torch.bfloat16, device='cuda')
    model.initialize_vision_modules()
    model.cuda()
    model.set_vision_tower_dtype(torch.bfloat16)
    model.print_device_configuration()

    tokenizer = transformers.AutoTokenizer.from_pretrained('kevin510/friday')

    # ───────── freeze LLM & tower; leave adapter trainable ─────────
    model.set_language_model_requires_grad(False)
    model.set_vision_tower_requires_grad(False)
    model.set_vision_adapter_requires_grad(True)

    optim = torch.optim.AdamW(model.model.mm_projector.parameters(), lr=0.0003)

    # ───────── build one synthetic batch (you can load a real datum instead) ────────
    dataset = PretrainingDataset(
        data_path="./tests/assets/data.json",
        image_dir="./tests/assets/images/",
        tokenizer=tokenizer,
        vision_tower=model.get_vision_tower(),
    )

    collator = PretrainingCollator(
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
