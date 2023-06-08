import os

import PIL
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer

from src import myclip as clip


class ConCapDataset(Dataset):
    def __init__(self, split, img_encoder_name, text_encdec_name, device='cuda', subset_size=None):
        # data
        self.data = pd.read_csv(f'/projects/datasets/concap/{split}.tsv', sep='\t', header=None)
        self.img_folder_path = f'/userhomes/yejoon/tl_summer2023/images/{split}'

        # for preprocessing
        _, self.img_preprocess = clip.load(img_encoder_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encdec_name)

        # if subset_size is set, use only the first subset_size samples
        self.subset_size = subset_size

    def __len__(self):
        if self.subset_size is None or self.subset_size > len(self.data):
            return len(self.data)
        else:
            return self.subset_size

    def _preprocess_image(self, img: Image) -> torch.Tensor:
        """Preprocess raw image into tensor"""
        # encode image
        img_input = self.img_preprocess(img).unsqueeze(0)  # size = (B, C, W, H) = (*, 3, 224 ,224)
        return img_input

    def __getitem__(self, index):
        caption = self.data.iloc[index, 0]

        img_path = os.path.join(self.img_folder_path, f"{index}.jpg")
        if not os.path.exists(img_path):
            # If the image file does not exist, return None.
            return None

        try:
            img = Image.open(img_path)
        except PIL.UnidentifiedImageError:
            # returning None causes error when using num_workers > 1
            # instead, return a different random sample
            # potentially causes infinite loop, but it's highly unlikely
            rand_i = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(rand_i)

        img_input = self._preprocess_image(img)

        return caption, img_input  # returning raw caption instead of tokenized one

    def collate_fn(self, batch):
        # remove None samples, this will result in a smaller batch size
        batch = list(filter(lambda x: x is not None, batch))
        captions, images = zip(*batch)

        # tokenize and pad captions to the longest one in the batch
        captions = self.tokenizer(captions, padding="longest", return_tensors="pt").input_ids

        # concatenate images along a new dimension
        images = torch.cat(images)

        return captions, images


# test code
if __name__ == '__main__':
    dataset = ConCapDataset('train', "ViT-B/16", "google/flan-t5-base")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=dataset.collate_fn)

    for captions, images in dataloader:
        print(captions.size())
        print(images.size())
        break
