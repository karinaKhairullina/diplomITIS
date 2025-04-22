import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import joblib
from anomaly_detection.config import DATA_DIR, MODELS_DIR, DATASET_FILES, ISOLATION_FOREST_PARAMS


def load_and_prepare_data(file_path):
    """
    Загружает и подготавливает данные: выбирает числовые признаки и заполняет пропуски.
    """
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=['int64', 'float64'])  # Только числовые данные
    df.fillna(df.mean(), inplace=True)  # Обработка пропусков
    return df


def get_feature_names_list(base_dir, dataset_files):
    """
    Автоматически определяет список признаков для каждого датасета.
    """
    feature_names_list = []
    for file in dataset_files:
        file_path = os.path.join(base_dir, file)
        df = load_and_prepare_data(file_path)
        feature_names_list.append(df.columns.tolist())
    return feature_names_list

def train_and_save_models():
    """
    Обучает и сохраняет модели Isolation Forest для каждого признака.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    feature_names_list = get_feature_names_list(DATA_DIR, DATASET_FILES)

    for i, file in enumerate(DATASET_FILES):
        file_path = os.path.join(DATA_DIR, file)
        df = load_and_prepare_data(file_path)

        for feature in df.columns:

            scaler = RobustScaler()
            scaled_feature = scaler.fit_transform(df[[feature]])

            model = IsolationForest(**ISOLATION_FOREST_PARAMS)
            model.fit(scaled_feature)


            preds = model.predict(scaled_feature)
            n_anomalies = (preds == -1).sum()

            print(f"Модель {i + 1} обучена на ({file}) для признака {feature}: {n_anomalies} аномалий")

            # Сохраняем модель и scaler
            joblib.dump(model, os.path.join(MODELS_DIR, f'model_{i}_{feature}.joblib'))
            joblib.dump(scaler, os.path.join(MODELS_DIR, f'scaler_{i}_{feature}.joblib'))


if __name__ == "__main__":
    train_and_save_models()