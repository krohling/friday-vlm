import json
import random
import transformers
from friday.data.finetuning import preprocess_for_finetuning
from friday.model.vision_tower import FastVitVisionTowerS2

NUM_SAMPLES = 1000

dataset_path = "./datasets/llava_v1_5_mix665k_small/llava_v1_5_mix665k_small.json"
images_path = "./datasets/llava_v1_5_mix665k_small/"

# open the dataset file
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

print(f"Loaded dataset with {len(dataset)} samples.")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "kevin510/friday", 
    trust_remote_code=True,
    use_fast=True,
    max_length=8192,
    padding_side="right",
)
print(f"tokenizer.model_max_length: {tokenizer.model_max_length}")
if tokenizer.unk_token is not None and tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

vision_tower = FastVitVisionTowerS2(
    pretrained_model_name_or_path="kevin510/fast-vit-hd", 
    s2_scales="512,1024", 
    model_params={"trust_remote_code": True}
)

sample_lengths = []
for i in range(NUM_SAMPLES):
    print(f"Processing sample {i + 1}/{NUM_SAMPLES}...")
    # randomly select a sample from the dataset (dont worry about duplicates)
    sample = dataset[random.randint(0, len(dataset) - 1)]
    processed_sample = preprocess_for_finetuning(
        sample,
        images_path,
        vision_tower,
        tokenizer
    )
    length = len(processed_sample['input_ids']) + len(processed_sample['labels']) + vision_tower.num_patches
    sample_lengths.append(length)

# count how many are longer than the tokenizer's max length
longer_than_max_length = sum(1 for length in sample_lengths if length > tokenizer.model_max_length)
print(f"Number of samples longer than tokenizer's max length ({tokenizer.model_max_length}): {longer_than_max_length}")


# Plot distribution of sample lengths using matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(sample_lengths, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Sample Lengths')
plt.xlabel('Sample Length')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('sample_length_distribution.png')