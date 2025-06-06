from friday.model import FridayForCausalLM        # your subclass
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch, os

# -----------------------------------------------------------
# 1) 4-bit quantisation config
# -----------------------------------------------------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,                  # ← turn on 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # mat-mul in bf16 (or fp16/fp32)
    bnb_4bit_quant_type="nf4",          # NF4 is the default & best
    bnb_4bit_use_double_quant=True,     # second-level quant improves quality
)

# -----------------------------------------------------------
# 2) load the merged dense checkpoint *with* that config
# -----------------------------------------------------------
quant_model = FridayForCausalLM.from_pretrained(
    "./model/friday",           # your merged fp16/bf16 model folder
    device_map="auto",          # bnb will shard automatically
    quantization_config=bnb_cfg,
    trust_remote_code=True,     # keep because FridayForCausalLM is custom
)
quant_model.eval()

# -----------------------------------------------------------
# 3) save the 4-bit checkpoint (same size as now, ~25% of fp16)
# -----------------------------------------------------------
export_dir = "./friday_int4"
os.makedirs(export_dir, exist_ok=True)
quant_model.save_pretrained(export_dir, safe_serialization=True)

print("✓ exported 4-bit NF4 model to", export_dir)

# -----------------------------------------------------------
# 4) sanity-check reload (CPU works too)
# -----------------------------------------------------------
from transformers import AutoModelForCausalLM
check = AutoModelForCausalLM.from_pretrained(
    export_dir,
    device_map="cpu",
)
print("dtype of first param:", next(check.parameters()).dtype)   # int8 (uint4 packed)
