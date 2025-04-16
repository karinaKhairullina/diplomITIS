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
    'data_group1.csv',
    'data_group2.csv',
    'data_group3.csv',
    'end_data4.csv',
    'end_data5.csv',
    'end_data6.csv',
    'end_data7.csv',
    'end_data8.csv',
    'end_data9.csv',
]

# Словарь альтернативных названий для каждого признака
alternative_names = {
    "Kills": ["HK", "Elim", "Kills", "KB"],
    "Deaths": ["D", "Deaths"],
    "Dmg": ["DD", "ADR"],
}

# Маппинг признаков на номера датасетов
FEATURE_INDEX_MAPPING = {}

# Формируем маппинг признаков на основе номеров датасетов
for i, dataset_file in enumerate(DATASET_FILES):
    file_path = os.path.join(DATA_DIR, dataset_file)
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        continue

    df = pd.read_csv(file_path)
    features = df.columns.tolist()

    # Добавляем признаки в FEATURE_INDEX_MAPPING с номером датасета
    for feature in features:
        FEATURE_INDEX_MAPPING[feature] = i

# Словарь для маппинга альтернативных названий к нормализованным признакам
ALTERNATIVE_NAMES = {}

# Создаём ALTERNATIVE_NAMES, где каждому альтернативному названию соответствует нормализованный признак
for normalized, aliases in alternative_names.items():
    for alias in aliases:
        ALTERNATIVE_NAMES[alias] = normalized

# Параметры модели
ISOLATION_FOREST_PARAMS = {
    'contamination': 0.03,
    'random_state': 42
}

# Проверка результата
print("ALTERNATIVE_NAMES:", ALTERNATIVE_NAMES)
