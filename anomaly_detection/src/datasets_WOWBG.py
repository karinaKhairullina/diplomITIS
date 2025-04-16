import pandas as pd
import os

# Определяем базовую директорию
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data', 'raw')
output_dir = os.path.join(base_dir, '..', 'data', 'processed')

# Создаем выходную директорию, если она не существует
os.makedirs(output_dir, exist_ok=True)

required_files = ["wowbgs.csv", "wowgil.csv", "wowsm.csv", "wowtk.csv", "wowwg.csv"]

# Словарь для хранения данных по группам
data_dict = {
    'data_group1': [],
    'data_group2': [],
    'data_group3': [],
    'end_data8': []
}

for file_name in required_files:
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Выбираем нужные столбцы
        columns_needed = ['KB', 'D', 'HK', 'DD', 'HD']
        existing_columns = [col for col in columns_needed if col in df.columns]

        df_selected = df[existing_columns]

        # Проверка на пропущенные значения (NaN)
        if df_selected.isna().sum().any():
            # Удаление строк с пропущенными значениями
            df_selected = df_selected.dropna()

        # Разделяем данные по группам
        if 'KB' in df_selected.columns and 'HK' in df_selected.columns:
            data_dict['data_group1'].append(df_selected[['KB', 'HK']])
        if 'D' in df_selected.columns:
            data_dict['data_group2'].append(df_selected[['D']])
        if 'DD' in df_selected.columns:
            data_dict['data_group3'].append(df_selected[['DD']])
        if 'HD' in df_selected.columns:
            data_dict['end_data8'].append(df_selected[['HD']])

# Функция для сохранения данных в файл
def save_data(data_list, output_file, description):
    """Добавляет новый столбец к существующему CSV, или создает новый файл."""
    if data_list:
        new_df = pd.concat(data_list, ignore_index=True)

        # Если файл уже существует, читаем и добавляем к нему
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)

            # Объединяем по столбцам (горизонтально), предполагая одинаковое число строк
            combined_df = pd.concat([existing_df, new_df.reset_index(drop=True)], axis=1)
        else:
            combined_df = new_df

        # Сохраняем в файл, заменяя старый
        combined_df.to_csv(output_file, index=False)
    else:
        print(f"⚠️ Нет данных для {description}.")



# Сохраняем данные для каждой группы в отдельный файл
save_data(data_dict['data_group1'], os.path.join(output_dir, 'data_group1.csv'), "Группа 1")
save_data(data_dict['data_group2'], os.path.join(output_dir, 'data_group2.csv'), "Группа 2")
save_data(data_dict['data_group3'], os.path.join(output_dir, 'data_group3.csv'), "Группа 3")
save_data(data_dict['end_data8'], os.path.join(output_dir, 'end_data8.csv'), "Одиночные признаки")