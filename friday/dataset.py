import json
import zipfile
import requests
from io import BytesIO
from PIL import Image
from s2wrapper import forward as multiscale_forward
from torchvision import transforms



def fn_pretrain_collate(batch):
    captions, images = zip(*batch)
    return list(captions), list(images)


def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

class PretrainDataset:

    def __init__(self, 
                 metadata_path, 
                 images_path, 
                 max_count=None
        ):
        """
        Initialize the PretrainDataset with paths to chat, metadata, and images.

        Args:
            metadata_path (str): Path to the metadata file.
            images_path (str): Path to the images directory.
        """
        self.metadata_path = metadata_path
        self.images_path = images_path
        self.transform = transforms.ToTensor()

        # load the metadata json file
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        # limit the number of metadata entries if max_count is specified
        if max_count is not None:
            self.metadata = self.metadata[:max_count]

        self.images_zip_file = zipfile.ZipFile(self.images_path, 'r')
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        metadata_entry = self.metadata[idx]

        caption = metadata_entry['blip_caption'] if metadata_entry['blip_caption'] else metadata_entry['caption']

        # load original image using 'url' attribute
        image = load_image_from_url(metadata_entry['url'])
        if image is not None:
            # return caption, self.transform(image)
            return caption, image
        

        with self.images_zip_file.open(metadata_entry['image']) as img_file:
            image_data = BytesIO(img_file.read())
            image = Image.open(image_data).convert("RGB")
            # return caption, self.transform(image)
            return caption, image
