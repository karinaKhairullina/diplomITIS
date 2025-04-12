import pandas as pd
import os
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'raw')
output_dir = os.path.join(base_dir, '..', 'processed')

required_files = ['all_seasons.csv']

final_df = []

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

        # Генерация синтетических данных
        num_new_rows = 200000
        new_data = []

        for _ in range(num_new_rows):
            new_sr_change = df_cleaned['SR Change'].sample(1).values[0] + np.random.randint(-5, 6)
            new_elim = df_cleaned['Elim'].sample(1).values[0] + np.random.randint(-2, 3)
            new_death = df_cleaned['Death'].sample(1).values[0] + np.random.randint(-1, 2)
            new_match_time = df_cleaned['Match Time'].sample(1).values[0] + np.random.randint(-1, 2)
            new_dmg = df_cleaned['Dmg'].sample(1).values[0] + np.random.randint(-500, 500)

            new_data.append([new_sr_change, new_elim, new_death, new_match_time, new_dmg])

        new_df = pd.DataFrame(new_data, columns=['SR Change', 'Elim', 'Death', 'Match Time', 'Dmg'])
        expanded_df = pd.concat([df_cleaned, new_df], ignore_index=True)

        final_df.append(expanded_df)

if final_df:
    final_df = pd.concat(final_df, ignore_index=True)
    output_file = os.path.join(output_dir, 'end_data2.csv')
    final_df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.2f')

