import joblib
import numpy as np
from anomaly_detection.config import REQUIRED_FEATURES
from django.conf import settings
from pathlib import Path


PROJECT_ROOT = Path(settings.BASE_DIR).parent
MODELS_DIR = PROJECT_ROOT / 'anomaly_detection' / 'models'
scaler = joblib.load(MODELS_DIR / 'scaler_data.joblib')
model = joblib.load(MODELS_DIR / 'model_data.joblib')


def prepare_test_data(df, required_columns):
    df_processed = df.copy()

    # 1) Убедиться, что все нужные колонки существуют
    for col in [c for c in required_columns if not c.endswith('_missing')]:
        if col not in df_processed.columns:
            df_processed[col] = np.nan  # создаём колонку с NaN

    # 2) Для каждой колонки ставим флаг по-строчно и заполняем NaN=0
    for col in [c for c in required_columns if not c.endswith('_missing')]:
        miss_col = f"{col}_missing"
        # флаг = 1 там, где было NaN, иначе 0
        df_processed[miss_col] = df_processed[col].isna().astype(int)
        # сам признак: NaN → 0
        df_processed[col] = df_processed[col].fillna(0)

    # 3) Гарантируем порядок и отсутствие лишних колонок
    df_processed = df_processed.reindex(columns=required_columns, fill_value=0)
    return df_processed


def analyze_data(df):
    df_prepared = prepare_test_data(df, REQUIRED_FEATURES)
    df_scaled = scaler.transform(df_prepared)
    preds = model.predict(df_scaled)
    n_anomalies = (preds == -1).sum()
    return preds, n_anomalies


def analyze_data_chunk(df_chunk):
    """
    Анализ небольшого куска датафрейма (для параллельной обработки).
    """
    df_prepared = prepare_test_data(df_chunk, REQUIRED_FEATURES)
    df_scaled = scaler.transform(df_prepared)
    preds = model.predict(df_scaled)
    return preds.tolist()


