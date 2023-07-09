import os
from dataclasses import dataclass
from typing import Optional

import PIL
from PIL import Image
import torch
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src import myclip as clip


class ImageCaptioner(torch.nn.Module):
    def __init__(self, img_encoder_name: str, text_encdec_name: str, lora_config: Optional[dict] = None, device='cuda'):
        """
        Initialize ImageCaptioner.
        :param img_encoder_name (str): img_encoder_name for OpenAI CLIP. Used in `clip.load(img_encoder_name, device=device)`
        :param text_encdec_name (str): text_encdec_name for 🤗 transformers. Used in `AutoModelForSeq2SeqLM.from_pretrained(text_encdec_name)`
        :param lora_config (dict, optional): lora_config for 🤗 peft. Should include args for `peft.LoraConfig`
        :param device (optional): torch device to use. Default: 'cuda'
        """
        super().__init__()

        # device
        self.device = device

        # image encoder (CLIP)
        self.img_encoder, self.img_preprocess = clip.load(img_encoder_name, device=device)

        # text encoder-decoder (flan-t5)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encdec_name)
        self.text_encdec = AutoModelForSeq2SeqLM.from_pretrained(text_encdec_name).to(device)

        # projection matrix; CLIP hidden dim -> T5 hidden dim (512 -> 768)
        self.projection = torch.nn.Linear(512, self.text_encdec.model_dim).to(device)

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
        img_features = self.projection(img_features)  # size = (B, Number of patches, T5 hidden dim) = (*, 196, 768)

        # text encoder-decoder
        loss = self.text_encdec(encoder_outputs=(img_features,), labels=tokenized_caption).loss  # size = (B, L, T5 vocab size) = (*, L, 32128)
        return loss

    def save(self, save_dir: str):
        """Only save the projection and lora parameters. Create save_dir if not exists."""
        # create save_dir if not exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # projection
        torch.save(self.projection.state_dict(), save_dir + "/projection.pt")

        # lora (this only saves the lora parameters)
        self.text_encdec.save_pretrained(save_dir)

    def load(self, save_dir: str):
        """Only load the trained parameters: projection and lora parameters. The rest should be loaded from the pretrained model."""
        # projection
        self.projection.load_state_dict(torch.load(save_dir + "/projection.pt", map_location=self.device))

        # lora
        self.text_encdec = PeftModel.from_pretrained(self.text_encdec, save_dir)

    @torch.no_grad()
    def generate_caption(self, img: PIL.Image) -> str:
        # preprocess and encode image
        img_input = self.img_preprocess(img).unsqueeze(0).to(self.device)  # size = (*, 3, 224, 224)
        img_features = self.img_encoder.encode_image(img_input, get_seq=True)  # size = (*, 196, 512)
        img_features = img_features.type(self.text_encdec.dtype)
        img_features = self.projection(img_features)  # size = (*, 196, 768)

        # encoder_outputs has img_features as 'last_hidden_state' attribute
        # but also encoder_outputs[0] should be img_features
        @dataclass
        class EncoderOutput:
            last_hidden_state: torch.Tensor
            def __getitem__(self, index):
                if index == 0:
                    return self.last_hidden_state
                else:
                    raise IndexError('Index out of range. You can only access index 0.')
            def __len__(self):
                return 1

        encoder_outputs = EncoderOutput(last_hidden_state=img_features)

        # generate caption
        self.text_encdec.eval()
        output = self.text_encdec.generate(inputs=None, encoder_outputs=encoder_outputs)
        caption = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return caption


# test code
if __name__ == '__main__':
    # Test 1: model save and load
    model = ImageCaptioner("ViT-B/16", "google/flan-t5-base", lora_config={"r": 2, "lora_dropout": .1, "lora_alpha": 16, "target_modules": ['q'],},
                           device="cuda")

    model.save("../test_save")
    model.load("../test_save")

    # Test 2: model forward
    caption = torch.randint(0, 32128, (1, 17)).cuda()
    image = torch.rand((1, 3, 224, 224)).cuda()

    outputs = model(caption, image)
    print('Done!')
