from os.path import join
from os import listdir
import pandas as pd


def get_readers(file_path: str, data_path: str) -> pd.DataFrame:
    '''
    Функция загрузки информации о дикторах
    '''
    readers = pd.read_csv(file_path, sep='\t').reset_index()
    readers.columns = ['READER', 'GENDER', 'SUBSET', 'NAME']
    
    readers_in_data = list(map(int, listdir(data_path)))
    
    readers = readers.merge(pd.Series(readers_in_data, name='READER'), on='READER')
    readers.set_index('READER', inplace=True)
    return readers

def get_chapters_info(file_path: str) -> pd.DataFrame:
    '''
    Функция загрузки информации о главах
    '''
    with open(file_path, 'r') as f:
        raw_text = list(map(lambda x: x.strip('\n'), f.readlines()))
    
    # В начале файла содержится дополнительная информация, нам нужна только таблица после неё
    header_end = 0
    for i, line in enumerate(raw_text):
        if not line.startswith(';'):
            header_end = i
            break

    table = [map(str.strip, line.split('|')) for line in raw_text[header_end:]]
    columns = list(map(lambda x: x.strip(';').strip(), raw_text[header_end-1].split('|')))
    chapters_df = pd.DataFrame(table, columns=columns).astype(dtype={'ID':int, 'READER': int, 'MINUTES': float})
    return chapters_df

def collect_paths_with_meta(path: str, readers: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция для извлечения списка путей до файлов каждого из диктора
    '''
    paths_lst = []
    # Бежим по директории с данными, заходим в директорию к диктору
    for reader in listdir(path):
        chapters_path = join(path, reader)
        # Заходим в директорию с главой
        for chapter in listdir(chapters_path):
            files_path = join(chapters_path, chapter)
            # Смотрим на все файлы этой главы и этого диктора
            for file in listdir(files_path):
                # Нам интересны только .wav
                if file.endswith('.wav'):
                    # Извлекаем пол диктора
                    reader_gender = readers.loc[int(reader)].GENDER
                    paths_lst.append((int(reader), reader_gender, join(files_path, file)))
                     
    return pd.DataFrame(paths_lst, columns=['reader', 'gender', 'path'])



