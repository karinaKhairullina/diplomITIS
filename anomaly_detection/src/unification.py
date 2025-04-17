import pandas as pd
import os

# Каталоги
base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, '..', 'data', 'processed')
output_dir = os.path.join(base_dir, '..', 'data', 'processed')
os.makedirs(output_dir, exist_ok=True)

# Словарь: файл → финальное название столбца
group_files = {
    'data_group1.csv': 'Kills',
    'data_group2.csv': 'Deaths',
    'data_group3.csv': 'Dmg'
}

for file_name, final_column_name in group_files.items():
    file_path = os.path.join(input_dir, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Собираем все значения из всех колонок
        all_values = pd.concat([df[col] for col in df.columns], ignore_index=True)

        # Удаляем пустые значения и строки только с пробелами
        all_values = all_values.dropna()
        all_values = all_values[all_values.astype(str).str.strip() != ""]

        # Финальный DataFrame
        result_df = pd.DataFrame({final_column_name: all_values})

        # Сохраняем
        output_path = os.path.join(output_dir, file_name)
        result_df.to_csv(output_path, index=False)
        print(f"{file_name} → сохранён как один столбец '{final_column_name}' ({len(result_df)} строк)")
    else:
        print(f"Файл не найден: {file_name}")
