import torch
from torch import nn
from easydict import EasyDict
from dataset import PretrainDataset
from model.vision_encoder.siglip_encoder import SiglipVisionTowerS2

args = EasyDict({
    "vision_tower": "google/siglip2-so400m-patch14-384",
    # "vision_tower": "google/siglip2-base-patch16-384",
    "s2_scales": "384,768",
    "chat_path": "LLaVA-CC3M-Pretrain-595K/chat.json",
    "metadata_path": "LLaVA-CC3M-Pretrain-595K/metadata.json",
    "images_path": "LLaVA-CC3M-Pretrain-595K/images.zip",
    "dataset_max_count": None,  # Set to None to use all data
    "batch_size": 3,
    "num_workers": 0,  # Set to 0 for no additional workers
    "vision_tower": {
        "vision_tower": "google/siglip2-so400m-patch14-384",
        # "vision_tower": "google/siglip2-base-patch16-384",
        "s2_scales": "384,768",
    },
    "vision_adapter": {
        "input_dim": 2304,
        "hidden_dim": 512,
        "output_dim": 256,
        "layers": 2,
        "activation": "gelu"
    }
})



device = torch.device("mps" if torch.mps.is_available() else "cpu")

vision_tower = SiglipVisionTowerS2(args.vision_tower, args=args).to(device)
projection_mlp = nn.Sequential(
    nn.Linear(args.projection.input_dim, args.projection.hidden_dim),
    nn.GELU(),
    nn.Linear(args.projection.hidden_dim, args.projection.output_dim)
).to(device)

pretrain_datset = PretrainDataset(
    metadata_path=args.metadata_path,
    images_path=args.images_path,
    max_count=args.dataset_max_count,
)
data_loader = torch.utils.data.DataLoader(
    pretrain_datset, batch_size=5, shuffle=True, num_workers=0
)

import time
start_time = time.time()
batch_counter = 0
for captions, images in data_loader:
    print(f"Batch {batch_counter}:")
    images = images.to(device)
    img_features = vision_tower(images)
    print(img_features.shape)
    projection = projection_mlp(img_features)
    print(f"Projection shape: {projection.shape}")
    
    batch_counter += 1

print(f"Time taken: {time.time() - start_time:.2f} seconds")