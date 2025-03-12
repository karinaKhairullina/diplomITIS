import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'raw')
output_dir = os.path.join(base_dir, '..', 'processed')

required_files = ["wowbgs.csv", "wowgil.csv", "wowsm.csv", "wowtk.csv", "wowwg.csv"]

final_df = []

for file_name in required_files:
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        columns_needed = ['KB', 'D', 'HK', 'DD', 'HD']
        df_selected = df[columns_needed]

        # Проверка на пропущенные значения (NaN)
        if df_selected.isna().sum().any():
            # Удаление строк с пропущенными значениями
            df_selected = df_selected.dropna()

        final_df.append(df_selected)

if final_df:
    final_df = pd.concat(final_df, ignore_index=True)
    output_file = os.path.join(output_dir, 'end_data4.csv')
    final_df.to_csv(output_file, index=False)


