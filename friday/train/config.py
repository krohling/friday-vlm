import transformers
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FridayDataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    image_dir: Optional[str] = field(default=None)
    max_count: Optional[int] = field(default=None)


@dataclass
class FridayTrainingArguments(transformers.TrainingArguments):
    vision_adapter_lr: Optional[float] = None
    mpt_attn_impl: Optional[str] = field(default="triton")
    group_by_modality_length: bool = field(default=False)
    save_only_vision_adapter: bool = field(
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
    lora_enable: bool = field(default=False)
    lora_params: dict = field(default_factory=dict)
    bits_and_bytes_params: dict = field(default_factory=dict)
    freeze_language_model: bool = field(default=False)
    freeze_vision_tower: bool = field(default=False)
    freeze_vision_adapter: bool = field(default=False)
    
