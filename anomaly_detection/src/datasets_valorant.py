import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'raw')
output_dir = os.path.join(base_dir, '..', 'processed')

required_files = ['Game_Valorant.csv']

final_df = []

for file_name in required_files:
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        selected_columns = ['Kills', 'Deaths', 'ADR', 'Econ']
        df_selected = df[selected_columns]

        # Замена NaN на 0
        df_selected = df_selected.fillna(0)

        # Удаление строк, где есть хотя бы одно значение 0.0
        df_cleaned = df_selected[(df_selected != 0.0).all(axis=1)]
        final_df.append(df_cleaned)

if final_df:
    final_df = pd.concat(final_df, ignore_index=True)
    output_file = os.path.join(output_dir, 'end_data5.csv')
    final_df.to_csv(output_file, index=False)


