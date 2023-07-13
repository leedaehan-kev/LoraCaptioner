import os

import PIL
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer

from src import myclip as clip


class ConCapDataset(Dataset):
    """PyTorch Dataset for Conceptual Captions dataset"""
    def __init__(self, split, img_encoder_name, text_encdec_name, device='cuda', subset_size: int = -1):
        # data
        self.data = pd.read_csv(f'/projects/datasets/concap/{split}_valid.tsv', sep='\t', header=None)
        self.img_folder_path = f'/userhomes/yejoon/tl_summer2023/images/{split}'

        # for preprocessing
        _, self.img_preprocess = clip.load(img_encoder_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encdec_name)

        # if subset_size is set, use only the first subset_size samples
        # -1 means use all samples
        self.subset_size = subset_size

    def __len__(self):
        if 0 < self.subset_size < len(self.data):
            return self.subset_size
        else:
            return len(self.data)

    def _preprocess_image(self, img: Image) -> torch.Tensor:
        """Preprocess raw image into tensor"""
        # encode image
        img_input = self.img_preprocess(img).unsqueeze(0)  # size = (B, C, W, H) = (*, 3, 224 ,224)
        return img_input

    def __getitem__(self, index):
        caption = self.data.iloc[index, 1]
        img_idx = self.data.iloc[index, 0]

        img_path = os.path.join(self.img_folder_path, f"{img_idx}.jpg")
        img = Image.open(img_path)
        img_input = self._preprocess_image(img)

        return caption, img_input  # returning raw caption (str) instead of tokenized one (tensor)

    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        captions, images = zip(*batch)

        # tokenize and pad captions to the longest one in the batch
        # TODO: pad to the multiple of 8 in order to use mixed precision training
        captions = self.tokenizer(captions, padding="longest", return_tensors="pt").input_ids

        # concatenate images along a new dimension
        images = torch.cat(images)

        return captions, images


# test code
if __name__ == '__main__':
    dataset = ConCapDataset('train', "ViT-B/16", "google/flan-t5-base")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)

    for captions, images in dataloader:
        print(captions.size())
        print(images.size())
