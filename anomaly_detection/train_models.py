import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import joblib
from anomaly_detection.config import DATA_DIR, MODELS_DIR, ISOLATION_FOREST_PARAMS


def load_and_prepare_data(file_path):
    """
    Загружает и оставляет только числовые признаки.
    """
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=['int64', 'float64'])
    return df


def train_and_save_model():
    """
    Обучает и сохраняет модель Isolation Forest.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    file_path = os.path.join(DATA_DIR, 'train_data.csv')
    df = load_and_prepare_data(file_path)

    # Масштабирование данных
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df)

    # Обучение модели
    model = IsolationForest(**ISOLATION_FOREST_PARAMS)
    model.fit(scaled_data)

    # Предсказание и подсчёт аномалий
    preds = model.predict(scaled_data)
    n_anomalies = (preds == -1).sum()

    print(f"Модель обучена на {os.path.basename(file_path)}: найдено {n_anomalies} аномалий")

    # Пути сохранения
    model_path = os.path.join(MODELS_DIR, 'model_data.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler_data.joblib')

    # Сохраняем модель и скейлер
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


if __name__ == "__main__":
    train_and_save_model()

