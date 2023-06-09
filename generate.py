import yaml
from PIL import Image

from src.model import ImageCaptioner

CONFIG_PATH = './configs/default.yaml'
MODEL_CHECKPOINT = './model_checkpoints/abkx98e7'


# read config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# load model
model = ImageCaptioner(config['img_encoder_name'], config['text_encdec_name'], config['lora_config'], config['device'])
model.load(MODEL_CHECKPOINT)

# sample images from images/validation
x = 100
image_ids = [x+i for i in range(10)]

# generate captions
for i in image_ids:
    print(f'Generating caption for image {i}...')
    try:
        img = Image.open(f'./images/validation/{i}.jpg')
    except FileNotFoundError:
        print(f'Image {i} not found. Skipping...')
        continue

    caption = model.generate_caption(img)

    print(caption)

    # write to results dir
    with open(f'./results/{i}.txt', 'w') as f:
        f.write(caption)

    with open(f'./results/{i}.jpg', 'wb') as f:
        img.save(f)
