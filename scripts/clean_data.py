import os
import time

import PIL
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def validate_img(i):
    if not _is_img_valid(i):
        print(f"Invalid image: {i}")
        invalid_imgs.append(i)


def _is_img_valid(i):
    img_path = os.path.join(img_folder_path, f"{i}.jpg")
    if not os.path.exists(img_path):
        return False
    try:
        _ = Image.open(img_path)
        return True
    except PIL.UnidentifiedImageError:
        return False


if __name__ == '__main__':
    start = time.time()

    split = 'dev'
    data = pd.read_csv(f'/projects/datasets/concap/{split}.tsv', sep='\t', header=None)
    img_folder_path = f'/userhomes/yejoon/tl_summer2023/images/{split}'
    invalid_imgs = []

    # multithreading
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(validate_img, data.index)

    # save data after dropping invalid images
    data.drop(invalid_imgs, inplace=True)
    data.to_csv(f'/projects/datasets/concap/{split}_valid.tsv', sep='\t', header=None, index=True)

    print(f"The number of invalid images: {len(invalid_imgs)}")
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")

# Run details:
# train split:
# The number of invalid images: 649344
# Time elapsed: 1933.5218214988708 seconds
# dev split:
# The number of invalid images: 15840
# Time elapsed: 1.8891477584838867 seconds
