import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'raw')
output_dir = os.path.join(base_dir, '..', 'processed')

required_files = ['Everquest_data.csv']

final_df = []

for file_name in required_files:
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None)

        # Разделение строк на отдельные столбцы
        df = df[0].str.split(',', expand=True)

        columns = [
            'id', 'victim_id', 'victim_guild_id', 'victim_level',
            'attacker_id', 'attacker_guild_id', 'attacker_level',
            'zone_id', 'killed_at', 'killmail_raw_id'
        ]
        df.columns = columns

        # Очистка данных: удаление лишних кавычек
        df = df.apply(lambda x: x.str.strip('"""'))

        # Замена пустых значений и None на 0
        df = df.replace({'': 0, None: 0})

        # Удаление строк, где есть хотя бы один 0
        df = df[(df != 0).all(axis=1)]

        df['victim_level'] = pd.to_numeric(df['victim_level'], errors='coerce')
        df['attacker_level'] = pd.to_numeric(df['attacker_level'], errors='coerce')

        df = df.dropna(subset=['victim_level', 'attacker_level'])

        selected_columns = ['victim_level', 'attacker_level']
        final_df.append(df[selected_columns])

# Объединение всех данных
if final_df:
    final_df = pd.concat(final_df, ignore_index=True)

    # Генерация синтетических данных
    num_new_rows = 100000  # Количество новых строк
    new_data = []

    for _ in range(num_new_rows):
        new_victim_level = final_df['victim_level'].sample(1).values[0] + np.random.randint(-5, 6)
        new_attacker_level = final_df['attacker_level'].sample(1).values[0] + np.random.randint(-5, 6)

        # Добавление небольшого количества аномальных данных
        if np.random.rand() < 0.01:  # Аномалия с вероятностью 1%
            new_victim_level = np.random.randint(1, 100)  # Аномально низкий уровень
            new_attacker_level = np.random.randint(100, 200)  # Аномально высокий уровень

        new_data.append([new_victim_level, new_attacker_level])

    new_df = pd.DataFrame(new_data, columns=['victim_level', 'attacker_level'])

    expanded_df = pd.concat([final_df, new_df], ignore_index=True)

    output_file = os.path.join(output_dir, 'end_data3.csv')
    expanded_df.to_csv(output_file, index=False)
