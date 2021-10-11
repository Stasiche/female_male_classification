import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def calc_scores(y, preds):
# Вычисление метрик
    return {
        'accuracy': np.round(accuracy_score(y, preds), 3),
        'f1': np.round(f1_score(y, preds), 3),
        'matthews': np.round(matthews_corrcoef(y, preds), 3)
    }


def cross_val(clf, readers, dataset, kf, decomposer, ffi):
    # Функция подсчета кросс-валидации
    scores = {}
    # Для каждого трейн-вал разбиения индексов дикторов
    for train_reader_indxs, test_reader_indxs in kf.split(readers, readers.GENDER):
        # Преобразуем массив индексов в множество для быстрого поиска
        train_readers = set(readers.iloc[train_reader_indxs].index.values)
        test_readers  = set(readers.iloc[test_reader_indxs].index.values)

        # Разделяем датасет по дикторам на тренировочный и валидационный
        train_dataset = dataset[[el in train_readers for el in dataset.reader]]
        test_dataset  = dataset[[el in test_readers  for el in dataset.reader]]
        
        # Отделяем признаки в датасете от ненужного и целевой переменной
        X_train = train_dataset.iloc[:, ffi:].values
        y_train = train_dataset.gender.values

        X_test = test_dataset.iloc[:, ffi:].values
        y_test = test_dataset.gender.values
        
        # При необходимости понижаем размерность
        if decomposer is not None:
            X_train = decomposer.fit_transform(X_train)
            X_test = decomposer.transform(X_test)
        
        # Обучаем классификатор
        clf.fit(X_train, y_train)
        
        # Группируем метрики
        for score_name, score in calc_scores(y_test, clf.predict(X_test)).items():
            scores.setdefault(score_name, []).append(score)

    return scores, {score_name: np.round(np.mean(score),2) for score_name, score in scores.items()}
