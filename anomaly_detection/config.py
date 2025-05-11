import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'anomaly_detection', 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'anomaly_detection', 'models')


# Словарь альтернативных названий для каждого признака
FEATURE_ALIASES = {
    'Kills': ['kills', 'frags', 'killed_enemies'],
    'Death': ['death', 'deaths', 'fallen'],
    'Dmg': ['damage', 'dmg', 'total_damage'],
    'HK': ['honorable_kills', 'hk', 'honor_kills'],
    'HD': ['healing_done', 'heals', 'total_heal'],
    'SR Change': ['sr_change', 'sr_diff', 'sr_delta'],
    'Match Time': ['match_time', 'duration', 'game_time'],
    'Econ': ['economy', 'econ_score', 'money'],
    'timeSpentLocation': ['time_spent', 'location_time', 'time_in_zone'],
    'level_difference': ['level_diff', 'lvl_gap', 'difference_level'],
    'player_id': ['player_id', 'id', 'player']
}


REQUIRED_FEATURES = [
    'player_id',
    'timeSpentLocation',
    'level_difference',
    'SR Change',
    'Kills',
    'Death',
    'Dmg',
    'Match Time',
    'Econ',
    'HK',
    'HD',
    'timeSpentLocation_missing',
    'level_difference_missing',
    'SR Change_missing',
    'Kills_missing',
    'Death_missing',
    'Dmg_missing',
    'Match Time_missing',
    'Econ_missing',
    'HK_missing',
    'HD_missing'
]

# Параметры модели
ISOLATION_FOREST_PARAMS = {
    'contamination': 0.01,
    'random_state': 42,
    'n_estimators': 100
}

