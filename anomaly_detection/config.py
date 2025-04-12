# config.py

import os
import pandas as pd

# Определяем базовую директорию проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Путь к директории data/processed/
DATA_DIR = os.path.join(BASE_DIR, 'anomaly_detection', 'data', 'processed')

# Путь к моделям
MODELS_DIR = os.path.join(BASE_DIR, 'anomaly_detection', 'models')

# Список файлов
DATASET_FILES = [
    'end_data1.csv',
    'end_data2.csv',
    'end_data3.csv',
    'end_data4.csv',
    'end_data5.csv'
]

FEATURE_INDEX_MAPPING = {}

# Формируем маппинг признаков на основе номеров датасетов
for i, dataset_file in enumerate(DATASET_FILES):
    file_path = os.path.join(DATA_DIR, dataset_file)
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден. Пропускаем.")
        continue

    df = pd.read_csv(file_path)
    features = df.columns.tolist()

    # Добавляем признаки в FEATURE_INDEX_MAPPING с номером датасета
    for feature in features:
        FEATURE_INDEX_MAPPING[feature] = i

# Параметры модели
ISOLATION_FOREST_PARAMS = {
    'contamination': 0.03,
    'random_state': 42
}

# Порог аномалий
ANOMALY_THRESHOLD = 2  # Если хотя бы два признака аномальны - строка аномальна