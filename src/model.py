from typing import Optional

import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image

from src import myclip as clip


class ImageCaptioner(torch.nn.Module):
    def __init__(self, img_encoder_name: str, text_encdec_name: str, lora_config: Optional[dict] = None, device='cuda'):
        super().__init__()

        # device
        self.device = device

        # image encoder (CLIP)
        self.img_encoder, _ = clip.load(img_encoder_name, device=device)

        # text encoder-decoder (flan-t5)
        self.text_encdec = AutoModelForSeq2SeqLM.from_pretrained(text_encdec_name).to(device)

        # projection matrix; size = (CLIP dim, T5 dim) = (512, 768)
        self.projection = torch.nn.Parameter(torch.randn(512, self.text_encdec.model_dim, dtype=self.text_encdec.dtype)).to(device)

        # freeze img_encoder
        for param in self.img_encoder.parameters():
            param.requires_grad = False

        # lora on text encoder-decoder
        if lora_config is not None:
            self.text_encdec = get_peft_model(self.text_encdec, LoraConfig(**lora_config))

    def forward(self, tokenized_caption: torch.Tensor, img_input: torch.Tensor):
        """
        :param tokenized_caption: caption tokenized with tokenizer; size = (B, L)
        :param img_input: image input; size = (B, C, W, H) = (*, 3, 224, 224)
        :return:
        """
        # encode image
        img_features = self.img_encoder.encode_image(img_input, get_seq=True)  # size = (B, Number of patches, CLIP hidden dim) = (*, 196, 512)

        # project image features to match the dim of text decoder
        img_features = img_features.type(self.text_encdec.dtype)
        img_features = img_features @ self.projection  # size = (B, Number of patches, T5 hidden dim) = (*, 196, 768)

        # text encoder-decoder
        outputs = self.text_encdec(encoder_outputs=(img_features,), labels=tokenized_caption)  # size = (B, L, T5 vocab size) = (*, L, 32128)
        return outputs


# test code
if __name__ == '__main__':
    model = ImageCaptioner("ViT-B/16", "google/flan-t5-base", device="cuda")

    caption = "Some kind of caption"
    image = Image.open("../images/test.jpg")

    outputs = model(caption, image)
