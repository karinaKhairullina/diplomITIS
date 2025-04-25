import os
import pandas as pd


# Определяем базовую директорию проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Путь к директории data/processed/
DATA_DIR = os.path.join(BASE_DIR, 'anomaly_detection', 'data', 'processed')

# Путь к моделям
MODELS_DIR = os.path.join(BASE_DIR, 'anomaly_detection', 'models')


# Словарь альтернативных названий для каждого признака
alternative_names = {
    "SR Change": ["Change", "Rank Change", "Rating Change", "Skill Rating", "SR Delta"],
    "Kills": ["Elim", "Kill", "KB", "HK", "Eliminations", "Kill Count", "Kills Made"],
    "Deaths": ["Deaths", "D", "Death Count", "Killed", "Fallen"],
    "Match Time": ["Match", "Game Time", "Duration", "Match Duration", "Time Played"],
    "Dmg": ["DD", "ADR", "Damage", "Damage Dealt", "DPS", "Damage Output"],
    "level_difference": ["Level Diff", "Level Gap", "Rank Difference", "Skill Level Difference"],
    "HD": ["Healing", "Heal", "Health Restored", "Healing Done", "Healed"],
    "Econ": ["Economy", "Money Earned", "Earnings", "Player Economy", "Credits Earned"],
    "timeSpentLocation": ["timeLocation", "Time in Zone", "Location Time", "Zone Time", "Area Time"]
}

# Словарь для маппинга альтернативных названий к нормализованным признакам
ALTERNATIVE_NAMES = {}

# Создаём ALTERNATIVE_NAMES, где каждому альтернативному названию соответствует нормализованный признак
for normalized, aliases in alternative_names.items():
    for alias in aliases:
        ALTERNATIVE_NAMES[alias] = normalized

REQUIRED_FEATURES = [
    'player_id',
    'timeSpentLocation',
    'level_difference',
    'SR Change',
    'Kills',
    'Death',
    'Dmg',
    'Match Time','Econ','HK','HD','timeSpentLocation_missing','level_difference_missing',
    'SR Change_missing','Kills_missing','Death_missing','Dmg_missing','Match Time_missing','Econ_missing','HK_missing','HD_missing'

]


# Параметры модели
ISOLATION_FOREST_PARAMS = {
    'contamination': 0.02,
    'random_state': 42,
    'n_estimators': 200
}

