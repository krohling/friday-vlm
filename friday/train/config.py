import transformers
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)


# import accelerate
@dataclass
class FridayTrainingArguments(transformers.TrainingArguments):
    # distributed_state: Optional[accelerate.PartialState] = field(
    #     default=None, init=False, repr=False
    # )
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # device: str = field(default="auto")
    lazy_preprocess: bool = field(default=True)
    # remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    bits_and_bytes_params: dict = field(default_factory=dict)
    lora_params: dict = field(default_factory=dict)
    save_only_mm_projector: bool = field(
        default=False,
        metadata={
            "help":
                "If true, only save the multimodal projector. This is useful for training the projector separately."
        },
    )