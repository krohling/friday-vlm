# conda create -n friday-dist python=3.12
# conda activate friday-dist
# pip install --upgrade pip setuptools wheel build
# rm -rf dist
# python -m build
# pip install --upgrade dist/friday*.whl

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

REPO = "kevin510/friday"      # ← the un-quantised repo you pushed

# ► 1.  Load tokenizer (adds <image> etc. automatically)
tok = AutoTokenizer.from_pretrained(
    REPO,
    trust_remote_code=True,   # needed because you registered custom classes
)

# ► 2.  Load model on the Mac GPU (MPS) or CPU
device_map = {"": "mps"} if torch.backends.mps.is_available() else {"": "cpu"}

model = AutoModelForCausalLM.from_pretrained(
    REPO,
    device_map=device_map,
    torch_dtype=torch.float16  # MPS only supports fp16 / fp32
                               # On Intel Macs or CPU only: use torch.bfloat16 or torch.float32
                               # trust_remote_code pulls in FridayConfig/Model
    ,
    trust_remote_code=True,
)

model.eval()
print("✓ loaded:", model.__class__.__name__,
      "dtype:", next(model.parameters()).dtype,
      "device:", next(model.parameters()).device)

# ► 3.  Quick text-only poke (no image just to prove forward pass)
prompt = "You are Friday, an AI assistant.\nUser: Hello!\nFriday:"
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
    )

print(tok.decode(out[0], skip_special_tokens=True))
