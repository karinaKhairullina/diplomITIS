import pandas as pd
import os

# Определяем базовую директорию
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data', 'raw')
output_dir = os.path.join(base_dir, '..', 'data', 'processed')

# Создаем выходную директорию, если она не существует
os.makedirs(output_dir, exist_ok=True)

required_files = ['Game_Valorant.csv']

# Словарь для хранения данных по каждому столбцу
data_dict = {
    'Kills': [],
    'Deaths': [],  # Сохраняем в data_group2.csv
    'ADR': [],  # Сохраняем в data_group3.csv
    'Econ': []  # Сохраняем в end_data7.csv
}

for file_name in required_files:
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Выбираем нужные столбцы
        selected_columns = ['Kills', 'Deaths', 'ADR', 'Econ']
        existing_columns = [col for col in selected_columns if col in df.columns]

        df_selected = df[existing_columns]

        # Замена NaN на 0
        df_selected = df_selected.fillna(0)

        # Удаление строк, где есть хотя бы одно значение 0.0
        df_cleaned = df_selected[(df_selected != 0.0).all(axis=1)]

        # Разделяем данные по столбцам
        for column in existing_columns:
            data_dict[column].append(df_cleaned[[column]])

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



# Сохраняем данные для каждого столбца в отдельный файл
save_data(data_dict['Kills'], os.path.join(output_dir, 'data_group1.csv'), "Группа 1")
save_data(data_dict['Deaths'], os.path.join(output_dir, 'data_group2.csv'), "Группа 2")
save_data(data_dict['ADR'], os.path.join(output_dir, 'data_group3.csv'), "Группа 3")
save_data(data_dict['Econ'], os.path.join(output_dir, 'end_data7.csv'), "Одиночные признаки")