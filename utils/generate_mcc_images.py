from joblib import Parallel, delayed
import skimage.io
import librosa
import numpy as np
import pandas as pd
import os
from os.path import join


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, out, hop_length, n_mels):
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)

    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)
    img = 255-img 

    skimage.io.imsave(out, img)

def get_spectrogram(gender: str, path: str, mcc_images_path: str, hop_length: int):
    y, sr = librosa.load(path)
    file_name = f'{gender}_{os.path.basename(path).split(".")[0]}.png'
    spectrogram_image(y, sr, join(mcc_images_path, file_name), hop_length, 128)

def process_batch(meta_paths_batch: pd.DataFrame, mcc_images_path: str, hop_length: int):
    for _, (gender, path) in meta_paths_batch.iterrows():
        get_spectrogram(gender, path, mcc_images_path, hop_length)

def generate_mcc_images(mcc_images_path: str, meta_paths: pd.DataFrame, 
                        n_jobs: int, batch_size: int, hop_length:int):
    
    if not os.path.exists(mcc_images_path):
        os.mkdir(mcc_images_path)

    batch_number = int(np.ceil(len(meta_paths)/batch_size))
    print(f'batch number: {batch_number}')
    jobs = []
    for i in range(batch_number):
        batch = meta_paths[['gender', 'path']].iloc[i*batch_size:(i+1)*batch_size]
        jobs.append(delayed(process_batch)(batch, mcc_images_path, hop_length))
    Parallel(n_jobs=n_jobs, verbose=10)(jobs)