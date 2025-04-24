import json
import zipfile
import requests
import tqdm
from io import BytesIO
from PIL import Image
from s2wrapper import forward as multiscale_forward
from torchvision import transforms


def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        # print(f"Error during request: {e}")
        return None
    except Exception as e:
        # print(f"Error opening image: {e}")
        return None


META_DATA_PATH = './LLaVA-CC3M-Pretrain-595K/metadata.json'
IMAGES_PATH = './LLaVA-CC3M-Pretrain-595K/images.zip'
NEW_IMAGES_PATH = './LLaVA-CC3M-Pretrain-595K/new_images.zip'

with open(META_DATA_PATH, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

images_zip_file = zipfile.ZipFile(IMAGES_PATH, 'r')
new_images_zip_file = zipfile.ZipFile(NEW_IMAGES_PATH, 'w')

success_count = 0
# for metadata_item in metadata:
for metadata_item in tqdm.tqdm(metadata, desc="Processing images"):
    filename = metadata_item['image']
    url = metadata_item['url']

    image = load_image_from_url(url)
    if image is None:
        with images_zip_file.open(filename) as img_file:
            image_data = BytesIO(img_file.read())
            image = Image.open(image_data).convert("RGB")
    else:
        success_count += 1
    
    # save image to new zip file
    with BytesIO() as img_buffer:
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        new_images_zip_file.writestr(filename, img_buffer.read())

new_images_zip_file.close()
# Close the original zip file
images_zip_file.close()
# Close the new zip file

print(f"Successfully loaded {success_count} images from URLs out of {len(metadata)} total images.")