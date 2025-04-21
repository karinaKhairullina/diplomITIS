import os
import numpy as np
import pandas as pd


# Функция для генерации нормальных данных
def generate_normal_data(min_value, max_value, size):
    return np.random.uniform(min_value, max_value, size)


# Функция для создания тестовых данных
def generate_test_data(train_df, scenario=1, num_rows=1000):
    # Вычисление диапазонов значений для каждого признака на основе обучающих данных
    normal_ranges = {col: (train_df[col].min(), train_df[col].max()) for col in train_df.columns if
                     train_df[col].dtype != 'object'}

    # Генерация данных для каждого признака
    test_data = {}
    for feature, (min_value, max_value) in normal_ranges.items():
        normal_data = generate_normal_data(min_value, max_value, num_rows)

        # Добавляем данные в словарь
        test_data[feature] = normal_data  # Без аномальных данных

    # В зависимости от сценария, изменяем данные
    if scenario == 1:
        test_data['SR Change'] = np.random.choice(test_data['SR Change'], num_rows)
        test_data['level_difference'] = np.random.choice(test_data['level_difference'], num_rows)
        test_data['timeSpentLocation'] = np.random.choice(test_data['timeSpentLocation'], num_rows)
        test_data['Econ'] = np.random.choice(test_data['Econ'], num_rows)

    if scenario == 2:
        test_data['NoModel1'] = np.random.uniform(-10, 10, num_rows)
        test_data['Deaths'] = np.random.choice(test_data['Kills'], num_rows)
        test_data['Match Time'] = np.random.choice(test_data['Econ'], num_rows)

    if scenario == 3:
        test_data = {key: test_data[key] for key in list(test_data.keys())[:1]}

    if scenario == 4:
        kills_data = test_data['Kills']
        test_data = {
            'Kills': np.random.choice(kills_data, num_rows),
            'Elim': np.random.choice(kills_data, num_rows),
            'KB': np.random.choice(kills_data, num_rows),
            'HK': np.random.choice(kills_data, num_rows),
        }

    if scenario == 5:
        test_data['timeSpentLocation'] = np.random.choice([np.nan, 1, 2, 3], num_rows)
        test_data['level_difference'] = np.random.choice([np.nan, 4, 5, 6], num_rows)
        test_data['SR Change'] = np.random.choice([np.nan, 7, 8, 9], num_rows)

    if scenario == 6:
        dmg_data = test_data['Dmg']
        test_data = {
            'Dmg': np.random.choice(dmg_data, num_rows),
            'DD': np.random.choice(dmg_data, num_rows),
            'ADR': np.random.choice(dmg_data, num_rows),
            'DPS': np.random.choice(dmg_data, num_rows),
        }

    # Создаем DataFrame
    df_test = pd.DataFrame(test_data)

    # Добавляем смещения для RobustScaler (если нужно)
    for col in df_test.columns:
        df_test[col] += np.random.uniform(-1, 1, num_rows)

    return df_test


# Загрузка всех CSV файлов из папки
folder_path = "/Users/karina/Desktop/IsolationForest/anomaly_detection/data/processed"
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Загружаем все CSV файлы и объединяем их в один DataFrame
train_df = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in all_files], axis=1)

# Количество строк на каждый сценарий
scenario_row_counts = {
    1: 5000,
    2: 2000,
    3: 1000,
    4: 10000,
    5: 2000,
    6: 8000
}

# Генерация данных и сохранение в файлы
for scenario, num_rows in scenario_row_counts.items():
    test_df = generate_test_data(train_df, scenario=scenario, num_rows=num_rows)
    test_df.to_csv(f"test_data_scenario_{scenario}.csv", index=False)
