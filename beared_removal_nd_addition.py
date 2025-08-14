import pandas as pd
import shutil
import os
from tqdm import tqdm


df = pd.read_csv('Celeb_dataset/celeba_gender_beard.csv')


males = df[df['gender'] == 1]


domain_a = males[males['beard'] == 0]


domain_b = males[males['beard'] == 1]


os.makedirs('cycle_data/trainA', exist_ok=True)
os.makedirs('cycle_data/trainB', exist_ok=True)


def copy_images(df_subset, target_dir):
    for filename in tqdm(df_subset['filename']):
        src = f'Celeb_dataset/img_align_celeba/img_align_celeba/{filename}'
        dst = f'cycle_data/{target_dir}/{filename}'
        shutil.copy(src, dst)


copy_images(domain_a, 'trainA')
copy_images(domain_b, 'trainB')

