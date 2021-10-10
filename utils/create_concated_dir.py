import os
from os.path import join
from os import listdir
import pandas as pd
import numpy as np
from itertools import product

from joblib import Parallel, delayed

from utils.concat_audios import concat_audios

from typing import List, Tuple, Dict
import librosa
import soundfile as sf

def create_concated_dir(audio_path: str, concated_audios_path: str, sr: int):
    speakers = pd.read_csv('data/speakers.tsv', sep='\t').reset_index()
    speakers.columns = ['READER', 'GENDER', 'SUBSET', 'NAME']
    
    speakers_in_data = list(map(int, listdir(audio_path)))
    
    speakers = speakers.merge(pd.Series(speakers_in_data, name='READER'), on='READER')
    speakers.set_index('READER', inplace=True)
    
    with open('./data/CHAPTERS.txt', 'r') as f:
        raw_text = list(map(lambda x: x.strip('\n'), f.readlines()))

    header_end = 0
    for i, line in enumerate(raw_text):
        if not line.startswith(';'):
            header_end = i
            break

    table = [map(str.strip, line.split('|')) for line in raw_text[header_end:]]
    columns = list(map(lambda x: x.strip(';').strip(), raw_text[header_end-1].split('|')))
    chapters_df = pd.DataFrame(table, columns=columns).astype(dtype={'ID':int, 'READER': int, 'MINUTES': float})
    chapters_df.head()
    
    dataset_list = []
    for reader in listdir(audio_path):
        chapters_path = join(audio_path, reader)
        for chapter in listdir(chapters_path):
            files_path = join(chapters_path, chapter)
            for file in listdir(files_path):
                if file.endswith('.wav'):
                    reader_gender = speakers.loc[int(reader)].GENDER
                    dataset_list.append((int(reader), reader_gender, join(files_path, file)))
    dataset_df = pd.DataFrame(dataset_list, columns=['reader', 'gender', 'path'])
    dataset_df.head()
    
    paths_by_reader = dataset_df[['reader', 'path']].groupby('reader').agg(lambda x: list(x.values))

    concat_audios(paths_by_reader, concated_audios_path, sr, 60)
    
    
    
    
    
