import transformers
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)


# import accelerate
@dataclass
class FridayTrainingArguments(transformers.TrainingArguments):
    mm_projector_lr: Optional[float] = None
    mpt_attn_impl: Optional[str] = field(default="triton")
    group_by_modality_length: bool = field(default=False)
    save_only_mm_projector: bool = field(
        default=False,
        metadata={
            "help":
                "If true, only save the multimodal projector. This is useful for training the projector separately."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_params: dict = field(default_factory=dict)
    bits_and_bytes_params: dict = field(default_factory=dict)
    
