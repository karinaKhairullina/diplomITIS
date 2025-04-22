import os
import numpy as np
import pandas as pd


# Функция для генерации нормальных данных
def generate_normal_data(mean, std_dev, size):
    return np.random.normal(mean, std_dev, size)


# Добавим функции смещения и шума
def add_small_noise(data, noise_factor=0.03):
    """
    Добавляет небольшой шум к данным.
    """
    noise = np.random.normal(0, noise_factor * np.std(data), size=len(data))
    return data + noise


def apply_small_shift(data, shift_factor=0.5):
    """
    Смещает данные на небольшое значение от их среднего.
    """
    shift = shift_factor * np.mean(data)
    return data + shift


# Функция для генерации тестовых данных для одного признака
def generate_test_data_for_feature(file_path, feature, num_rows=1000):
    """
    Генерирует тестовые данные для одного признака, используя файл с данными.
    Добавляет шум и небольшое смещение для реалистичности.
    """
    train_df = pd.read_csv(file_path)

    mean = train_df[feature].mean()
    std_dev = train_df[feature].std()
    min_value = train_df[feature].min()
    max_value = train_df[feature].max()

    # Генерация данных
    normal_data = generate_normal_data(mean, std_dev, num_rows)

    # Добавляем шум и смещение
    normal_data = add_small_noise(normal_data, noise_factor=0.03)
    normal_data = apply_small_shift(normal_data, shift_factor=0.02)

    # Ограничиваем диапазон
    normal_data = np.clip(normal_data, min_value, max_value)

    return normal_data


# Функция для создания тестовых данных с учетом сценариев
def generate_test_data(train_df, scenario=1, num_rows=1000):
    normal_ranges = {
        col: (train_df[col].min(), train_df[col].max())
        for col in train_df.columns if train_df[col].dtype != 'object'
    }

    test_data = {}
    for feature, (min_value, max_value) in normal_ranges.items():
        mean = train_df[feature].mean()
        std_dev = train_df[feature].std()
        normal_data = generate_normal_data(mean, std_dev, num_rows)
        normal_data = np.clip(normal_data, min_value, max_value)
        test_data[feature] = normal_data

    # === СЦЕНАРИЙ 1: Только определенные признаки ===
    if scenario == 1:
        selected_features = ['SR Change', 'level_difference', 'timeSpentLocation', 'Econ']
        test_data = {key: test_data[key] for key in selected_features}

    # === СЦЕНАРИЙ 2: Другие признаки + NoModel1 ===
    elif scenario == 2:
        excluded_features = {'SR Change', 'level_difference', 'timeSpentLocation', 'Econ'}
        selected_features = [f for f in test_data if f not in excluded_features]
        selected_data = {key: test_data[key] for key in selected_features[:4]}  # Возьмём 4 других признака
        selected_data['NoModel1'] = np.random.uniform(-10, 10, num_rows)
        test_data = selected_data

    # === СЦЕНАРИЙ 3: Оставить только первый признак ===
    elif scenario == 3:
        test_data = {key: test_data[key] for key in list(test_data.keys())[:1]}

    # === СЦЕНАРИЙ 4: Основан на Kills ===
    elif scenario == 4:
        kills_data = test_data['Kills']
        test_data = {
            'Kills': np.random.choice(kills_data, num_rows),
            'Elim': np.random.choice(kills_data, num_rows),
            'KB': np.random.choice(kills_data, num_rows),
            'HK': np.random.choice(kills_data, num_rows),
        }

    # === СЦЕНАРИЙ 5: 6 признаков + NaN в случайных местах ===
    elif scenario == 6:
        # Загружаем данные из одного файла 'data_group3.csv'
        dmg_data = generate_test_data_for_feature(file_path_mapping['Dmg'], 'Dmg', num_rows)

        # Генерация данных для других признаков на основе 'Dmg' с добавлением шума или аномалий
        test_data = {
            'Dmg': dmg_data,
            'DD': np.random.choice(dmg_data, num_rows),
            'ADR': np.random.choice(dmg_data, num_rows),
            'DPS': np.random.choice(dmg_data, num_rows),
        }

        # Добавляем случайные аномалии
        # Добавляем небольшие выбросы в каждый признак
        for key in ['DD', 'ADR', 'DPS']:
            anomaly_indices = np.random.choice(num_rows, size=int(num_rows * 0.05),
                                               replace=False)  # 5% аномальных данных
            test_data[key][anomaly_indices] = test_data[key][
                                                  anomaly_indices] * 1.5  # Умножаем на 1.5 для создания аномалии



    # === СЦЕНАРИЙ 6: Производные от Dmg (все признаки из одного файла) ===
    elif scenario == 6:
        # Загружаем данные из одного файла 'data_group3.csv'
        dmg_data = generate_test_data_for_feature(file_path_mapping['Dmg'], 'Dmg', num_rows)

        # Для каждого признака (Dmg, DD, ADR, DPS) генерируем данные из того же файла
        test_data = {
            'Dmg': dmg_data,
            'DD': np.random.choice(dmg_data, num_rows),
            'ADR': np.random.choice(dmg_data, num_rows),
            'DPS': np.random.choice(dmg_data, num_rows),
        }

    df_test = pd.DataFrame(test_data)
    return df_test


# Загрузка всех CSV файлов из папки
folder_path = "/Users/karina/Desktop/IsolationForest/anomaly_detection/data/processed"
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Создаем маппинг между признаками и соответствующими файлами
file_path_mapping = {
    'Kills': os.path.join(folder_path, 'data_group1.csv'),
    'Deaths': os.path.join(folder_path, 'data_group2.csv'),
    'Dmg': os.path.join(folder_path, 'data_group3.csv'),
    'timeSpentLocation': os.path.join(folder_path, 'end_data4.csv'),
    'level_difference': os.path.join(folder_path, 'end_data5.csv'),
    'SR Change': os.path.join(folder_path, 'end_data6.csv'),
    'Econ': os.path.join(folder_path, 'end_data7.csv'),
    'HD': os.path.join(folder_path, 'end_data8.csv'),
    'Match Time': os.path.join(folder_path, 'end_data9.csv'),
}

# Количество строк на каждый сценарий
scenario_row_counts = {
    1: 10000,
    2: 10000,
    3: 10000,
    4: 10000,
    5: 10000,
    6: 10000
}

# Генерация данных и сохранение в файлы
for scenario, num_rows in scenario_row_counts.items():
    # Генерация данных для каждого признака
    all_test_data = {}

    for feature, file_path in file_path_mapping.items():
        # Генерация тестовых данных для каждого признака
        test_data = generate_test_data_for_feature(file_path, feature, num_rows)
        all_test_data[feature] = test_data

    # Создание DataFrame
    df_test = pd.DataFrame(all_test_data)

    # Применение модификации в зависимости от сценария
    df_test = generate_test_data(df_test, scenario=scenario, num_rows=num_rows)

    # Сохраняем данные в CSV файл
    df_test.to_csv(f"generated_test_data_scenario_{scenario}.csv", index=False)
