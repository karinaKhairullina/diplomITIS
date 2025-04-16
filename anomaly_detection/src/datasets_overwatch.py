import pandas as pd
import os
import numpy as np

# Определяем базовую директорию
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data', 'raw')
output_dir = os.path.join(base_dir, '..', 'data', 'processed')

# Создаем выходную директорию, если она не существует
os.makedirs(output_dir, exist_ok=True)

required_files = ['all_seasons.csv']

# Списки для хранения данных по каждому признаку
sr_change_df = []
match_time_df = []
elim_df = []
death_df = []
dmg_df = []

def save_data(data_list, output_file, description):
    """Добавляет данные в файл, не перезаписывая существующие."""
    if data_list:
        new_data = pd.concat(data_list, ignore_index=True)

        # Если файл уже существует — считываем и добавляем к нему
        if os.path.exists(output_file):
            existing_data = pd.read_csv(output_file)

            # Объединяем старые и новые данные
            combined_df = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined_df = new_data

        # Удаляем дубликаты строк (если необходимо)
        combined_df = combined_df.drop_duplicates()

        # Сохраняем итог
        combined_df.to_csv(output_file, index=False)
    else:
        print(f"Нет данных для {description}.")

for file_name in required_files:
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Удаление полностью пустых строк и столбцов
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')

        selected_columns = ['SR Change', 'Elim', 'Death', 'Match Time', 'Dmg']
        existing_columns = [col for col in selected_columns if col in df.columns]

        df_selected = df[existing_columns].copy()
        df_selected = df_selected.dropna(how='any')

        # Преобразуем 'Match Time'
        if 'Match Time' in df_selected.columns:
            df_selected['Match Time'] = df_selected['Match Time'].astype(str)
            df_selected['Match Time'] = df_selected['Match Time'].str.extract(r'(\d+):')[0]
            df_selected['Match Time'] = pd.to_numeric(df_selected['Match Time'], errors='coerce')

        df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

        # Удаление строк, где есть хотя бы один 0 в любом из столбцов
        df_cleaned = df_selected[(df_selected != 0).all(axis=1)]
        df_cleaned = df_cleaned.dropna(how='all')

        # Средние значения и стандартные отклонения для каждого столбца
        means = df_cleaned.mean()
        std_devs = df_cleaned.std()

        # Генерация синтетических данных
        num_new_rows = 200000
        new_data = []

        for _ in range(num_new_rows):
            new_sr_change = np.random.normal(loc=means['SR Change'], scale=std_devs['SR Change'])
            new_elim = np.random.normal(loc=means['Elim'], scale=std_devs['Elim'])
            new_death = np.random.normal(loc=means['Death'], scale=std_devs['Death'])
            new_match_time = np.random.normal(loc=means['Match Time'], scale=std_devs['Match Time'])
            new_dmg = np.random.normal(loc=means['Dmg'], scale=std_devs['Dmg'])

            # Аномалии с вероятностью 1% (генерация значений, отдалённых от среднего)
            if np.random.rand() < 0.01:  # Аномалии для всех признаков с вероятностью 2%
                new_sr_change = means['SR Change'] + np.random.choice([20, -20])  # Аномалии для SR Change
                new_elim = means['Elim'] + np.random.choice([20, -20])  # Аномалии для Elim
                new_death = means['Death'] + np.random.choice([10, -10])  # Аномалии для Death
                new_match_time = means['Match Time'] + np.random.choice([20, -20])  # Аномалии для Match Time
                new_dmg = means['Dmg'] + np.random.choice([1000, -1000])  # Аномалии для Dmg

            new_data.append([new_sr_change, new_elim, new_death, new_match_time, new_dmg])

        # Создаем DataFrame с синтетическими данными
        new_df = pd.DataFrame(new_data, columns=['SR Change', 'Elim', 'Death', 'Match Time', 'Dmg'])
        expanded_df = pd.concat([df_cleaned, new_df], ignore_index=True)

        # Разделяем данные
        sr_change_data = expanded_df[['SR Change']].copy()
        match_time_data = expanded_df[['Match Time']].copy()
        elim_data = expanded_df[['Elim']].copy()
        death_data = expanded_df[['Death']].copy()
        dmg_data = expanded_df[['Dmg']].copy()

        # Добавляем данные в соответствующие списки
        sr_change_df.append(sr_change_data)
        match_time_df.append(match_time_data)
        elim_df.append(elim_data)
        death_df.append(death_data)
        dmg_df.append(dmg_data)

# Сохраняем данные с использованием функции
save_data(sr_change_df, os.path.join(output_dir, 'end_data6.csv'), "Одиночный признак SR Change")
save_data(match_time_df, os.path.join(output_dir, 'end_data9.csv'), "Одиночный признак Match Time")
save_data(elim_df, os.path.join(output_dir, 'data_group1.csv'), "Группа 1 (Elim)")
save_data(death_df, os.path.join(output_dir, 'data_group2.csv'), "Группа 2 (Death)")
save_data(dmg_df, os.path.join(output_dir, 'data_group3.csv'), "Группа 3 (Dmg)")
