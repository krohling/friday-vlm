import torch
from easydict import EasyDict
from dataset import PretrainDataset, fn_pretrain_collate
from model.friday_arch import FridayMetaModel
from model.language_model.friday_phi import build_friday_phi

config = EasyDict({
    "dataset": {
        "metadata_path": "LLaVA-CC3M-Pretrain-595K/metadata.json",
        "images_path": "LLaVA-CC3M-Pretrain-595K/images.zip",
        "max_count": None,
    },
    "train": {
        "batch_size": 3,
        "num_workers": 0,
    },
    "vision_tower": {
        "vision_tower": "google/siglip2-base-patch16-384",
        # "vision_tower": "google/siglip2-so400m-patch16-384",
        "s2_scales": "384,768",
    },
    "vision_adapter": {
        "input_dim": 1536,
        "hidden_dim": 512,
        "output_dim": 256,
        "layers": 2,
        "activation": "gelu"
    },
})



device = torch.device("mps" if torch.mps.is_available() else "cpu")

# model, tokenizer = build_friday_phi(config)
# model = model.to(device)
# model.eval()  # Set the model to evaluation mode

model = FridayMetaModel(
    cfg_vision_adapter=config.vision_adapter,
    cfg_vision_tower=config.vision_tower
).to(device)

pretrain_datset = PretrainDataset(**config.dataset)
data_loader = torch.utils.data.DataLoader(
    pretrain_datset, 
    batch_size=config.train.batch_size, 
    num_workers=config.train.num_workers, 
    collate_fn=fn_pretrain_collate,
    shuffle=True
)

import time
start_time = time.time()
batch_counter = 0
for captions, images in data_loader:
    print(f"Batch {batch_counter}:")
    # images = images.to(device)
    img_features = model.encode_images(images)
    print(f"img_features.shape: {img_features.shape}")
    
    batch_counter += 1

print(f"Time taken: {time.time() - start_time:.2f} seconds")