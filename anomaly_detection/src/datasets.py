import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'raw')
output_dir = os.path.join(base_dir, '..', 'processed')

required_files = ['playersData_Warhammer.csv', 'playersDataWOW1.csv', 'playersDataWOW2.csv']

final_df = []

for file_name in required_files:
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        if 'locationStart' in df.columns and 'locationStop' in df.columns:
            # Преобразование времени
            df['locationStart'] = pd.to_datetime(df['locationStart'], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
            df['locationStop'] = pd.to_datetime(df['locationStop'], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')

            # Вычисление времени в локации (в минутах)
            df['timeSpentLocation'] = (df['locationStop'] - df['locationStart']).dt.total_seconds() / 60

            # Замена пустых значений и None на 0
            df = df.replace({'': 0, None: 0})

            # Удаление строк, где есть хотя бы один 0 в любом из столбцов
            df = df[(df != 0).all(axis=1)]

            final_df.append(df[['timeSpentLocation']])

if final_df:
    final_df = pd.concat(final_df, ignore_index=True)
    output_file = os.path.join(output_dir, 'end_data1.csv')
    final_df.to_csv(output_file, index=False)
