import joblib
import pandas as pd
import numpy as np
from anomaly_detection.config import REQUIRED_FEATURES


def prepare_test_data(df, required_columns):
    """
    Гарантирует наличие всех признаков и *_missing-флагов,
    даже если в исходных данных они отсутствуют.
    """
    df_processed = df.copy()

    # Добавляем отсутствующие признаки и их флаги
    for col in [c for c in required_columns if not c.endswith('_missing')]:
        if col not in df.columns:
            df_processed[col] = 0  # Значение по умолчанию
            df_processed[f"{col}_missing"] = 1  # Флаг отсутствия
        else:
            df_processed[f"{col}_missing"] = 0  # Флаг присутствия

    # Убедимся, что все колонки на месте и в правильном порядке
    df_processed = df_processed.reindex(columns=required_columns, fill_value=0)
    return df_processed


def analyze_data(df):
    # Загружаем scaler и модель
    scaler = joblib.load('/Users/karina/Desktop/IsolationForest/anomaly_detection/models/scaler_data.joblib')
    model = joblib.load('/Users/karina/Desktop/IsolationForest/anomaly_detection/models/model_data.joblib')

    # Подготавливаем данные: добавляем отсутствующие признаки и флаги
    df_prepared = prepare_test_data(df, REQUIRED_FEATURES)

    # Масштабируем
    df_scaled = scaler.transform(df_prepared)

    # Предсказываем
    preds = model.predict(df_scaled)
    n_anomalies = (preds == -1).sum()

    return preds, n_anomalies

