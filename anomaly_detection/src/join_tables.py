import pandas as pd
import os
from sklearn.model_selection import train_test_split

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_dir = os.path.join(project_root, "data", "raw")
processed_dir = os.path.join(project_root, "data", "processed")


def safe_read_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        header_line = f.readline()
        sep = ';' if header_line.count(';') > header_line.count(',') else ','
    df = pd.read_csv(filepath, sep=sep)
    return df


def compute_time_spent(df, start_col, stop_col):
    if start_col not in df.columns or stop_col not in df.columns:
        raise ValueError(f"Одного из столбцов: {start_col} или {stop_col} нет в данных.")
    df[start_col] = pd.to_datetime(df[start_col], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
    df[stop_col] = pd.to_datetime(df[stop_col], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
    return (df[stop_col] - df[start_col]).dt.total_seconds() / 60


# 1. WoW timeSpentLocation
dfs = []
for filename in ['playersDataWOW1.csv', 'playersDataWOW2.csv']:
    filepath = os.path.join(data_dir, filename)
    df = safe_read_csv(filepath)
    if 'locationStart' in df.columns and 'locationStop' in df.columns:
        df['timeSpentLocation'] = compute_time_spent(df, 'locationStart', 'locationStop')
        dfs.append(df[['timeSpentLocation']])
time_df = pd.concat(dfs, ignore_index=True)
time_df['player_id'] = range(1, len(time_df) + 1)

# 2. Everquest
df = pd.read_csv(os.path.join(data_dir, 'Everquest_data.csv'), quotechar='"')
df_split = df.iloc[:, 0].astype(str).str.split(',', expand=True)
df_split.columns = ['id', 'victim_id', 'victim_guild_id', 'victim_level', 'attacker_id',
                    'attacker_guild_id', 'attacker_level', 'zone_id', 'killed_at', 'killmail_raw_id']
df_split = df_split.apply(lambda x: x.str.strip('"""')).replace({'': 0, None: 0})
df_split['victim_level'] = pd.to_numeric(df_split['victim_level'], errors='coerce')
df_split['attacker_level'] = pd.to_numeric(df_split['attacker_level'], errors='coerce')
df_split['level_difference'] = df_split['attacker_level'] - df_split['victim_level']
df_split = df_split[['level_difference']]
df_split['player_id'] = range(1, len(df_split) + 1)

# 3. Overwatch
def parse_match_time(t):
    try:
        mins, secs = map(int, str(t).split(':'))
        return mins + secs / 60
    except:
        return None

season_df = pd.read_csv(os.path.join(data_dir, 'all_seasons.csv'))
season_df['Match Time'] = season_df['Match Time'].apply(parse_match_time)
season_df = season_df[['SR Change', 'Kills', 'Death', 'Dmg', 'Match Time']]
season_df['player_id'] = range(1, len(season_df) + 1)

# 4. Valorant
valorant_df = pd.read_csv(os.path.join(data_dir, 'Game_Valorant.csv'))
valorant_df = valorant_df[['Kills', 'Death', 'Dmg', 'Econ']]
valorant_df['player_id'] = range(1, len(valorant_df) + 1)

# 5. WoW
wow_files = ['wowbgs.csv', 'wowgil.csv', 'wowsm.csv', 'wowtk.csv', 'wowwg.csv']
wow_dfs = []
for filename in wow_files:
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    df_selected = df[['Kills', 'Death', 'HK', 'Dmg', 'HD']]
    wow_dfs.append(df_selected)
wow_df = pd.concat(wow_dfs, ignore_index=True)
wow_df['player_id'] = range(1, len(wow_df) + 1)


all_scenarios_df = pd.concat([time_df, df_split, season_df, valorant_df, wow_df], ignore_index=True)
all_scenarios_df = all_scenarios_df.sort_values(by='player_id')

numeric_columns = all_scenarios_df.select_dtypes(include='number').columns.drop('player_id', errors='ignore')
aggregated_df = all_scenarios_df.groupby('player_id')[numeric_columns].agg('median')

aggregated_df.columns = [f'{col}' for col in aggregated_df.columns]
aggregated_df.reset_index(inplace=True)

def prepare_train_data(df):
    df_processed = df.copy()
    for col in df_processed.columns:
        if col != 'player_id':
            df_processed[f'{col}_missing'] = df_processed[col].isna().astype(int)
            df_processed[col] = df_processed[col].fillna(0)
    return df_processed

def prepare_test_data(df):
    return df.copy()

train_df, test_df = train_test_split(aggregated_df, test_size=0.2, random_state=42)

train_df = prepare_train_data(train_df)
test_df = prepare_test_data(test_df)

train_df.to_csv(os.path.join(processed_dir, 'train_data.csv'), index=False)
test_df.to_csv(os.path.join(processed_dir, 'test_data.csv'), index=False)
