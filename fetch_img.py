import requests
import io
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

SPLIT = 'train'


def fetch_single_image(url, i, timeout=10, retries=2):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # bytes to image
        img = Image.open(io.BytesIO(response.content))

        # save image
        img.save(f'./images/{SPLIT}/{i}.jpg')

    except Exception as e:
        # append to error log
        with open(f'./images/{SPLIT}/error.log', 'a') as f:
            f.write(f'{i}\t{url}\t{e}\n')


if __name__ == '__main__':
    path = f'/projects/datasets/concap/{SPLIT}.tsv'
    df = pd.read_csv(path, sep='\t', header=None)

    with ThreadPoolExecutor(max_workers=32) as executor:
        # create a list of tuples where each tuple is (url, index)
        url_and_indexes = list(zip(df[1], df.index))
        # executor.map will apply the function to every tuple in url_and_indexes
        executor.map(lambda pair: fetch_single_image(*pair), url_and_indexes)
