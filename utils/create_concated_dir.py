import pandas as pd
import numpy as np
import librosa
import os
from os.path import join
from utils.main_utils import get_readers, get_chapters_info, collect_paths_with_meta
from joblib import Parallel, delayed
import soundfile as sf


def process_batch(paths_by_reader_batch: pd.DataFrame, save_path: str, sr: int, duration_limit: int):
    '''
    Функция для обработки одного батча дикторов.
    Для каждого диктора из paths_by_reader, записи жадно склееваются на файлы длинной около duration_limit секунд длинной и записываются на диск в save_path. 

    '''
    for reader, paths in paths_by_reader_batch.itertuples():
        reader_path_save = join(save_path, str(reader))
        
        buffer = []
        flash_cnt, buffer_duration = 0, 0
        for path in paths:
            y = librosa.load(path)[0]
            buffer.append(y)
            buffer_duration += librosa.get_duration(y, sr)
            
            if buffer_duration >= duration_limit:
                sf.write(join(save_path, f'{reader}_{flash_cnt}.wav'), np.hstack(buffer), sr)
                
                flash_cnt += 1
                buffer = []
                buffer_duration = 0
        
        


def concat_audios(paths_by_reader: pd.DataFrame, concated_audios_path: str, sr: int, duration_limit: int,
                  n_jobs:int = 12, batch_size:int = 3): 
    ''''
    Функция для запуска склеевания
    '''

    if not os.path.exists(concated_audios_path):
        os.makedirs(concated_audios_path)

    batch_number = int(np.ceil(len(paths_by_reader)/batch_size))
    print(f'batch number: {batch_number}')
    jobs = []
    for i in range(batch_number):
        batch = paths_by_reader.iloc[i*batch_size:(i+1)*batch_size]
        jobs.append(delayed(process_batch)(batch, concated_audios_path, sr, duration_limit))
    Parallel(n_jobs=n_jobs, verbose=10)(jobs)
    

def create_concated_dir(audio_path: str, concated_audios_path: str, sr: int):
    readers = get_readers('data/speakers.tsv', audio_path)
    chapter_info = get_chapters_info('./data/CHAPTERS.txt')
    meta_paths = collect_paths_with_meta(audio_path, readers)
    
    paths_by_reader = meta_paths[['reader', 'path']].groupby('reader').agg(lambda x: list(x.values))

    concat_audios(paths_by_reader, concated_audios_path, sr, 60)
    
    
    
    
    
