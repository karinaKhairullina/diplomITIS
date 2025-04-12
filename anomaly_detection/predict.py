import os
import joblib
import pandas as pd
from anomaly_detection.config import FEATURE_INDEX_MAPPING, ANOMALY_THRESHOLD

class CombinedModel:
    def __init__(self, models_dir, feature_names_list):
        self.models_dir = models_dir
        self.feature_names_list = [feature.strip() for feature in feature_names_list]  # Убираем лишние пробелы
        self.models = {}
        self.scalers = {}
        self.unprocessed_features = []

        # Загрузка всех моделей и масштабировщиков для каждого признака
        for feature in self.feature_names_list:
            # Получаем индекс из маппинга
            i = FEATURE_INDEX_MAPPING.get(feature)

            if i is None:
                self.unprocessed_features.append(feature)
                continue

            model_path = os.path.join(models_dir, f'model_{i}_{feature}.joblib')
            scaler_path = os.path.join(models_dir, f'scaler_{i}_{feature}.joblib')

            # Проверяем, существуют ли файлы модели и масштабировщика
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    self.models[(i, feature)] = model
                    self.scalers[(i, feature)] = scaler
                    print(f"Модель для признака {feature}  загружена")
                except Exception as e:
                    print(f"Ошибка при загрузке модели для признака {feature}: {str(e)}")
                    self.unprocessed_features.append(feature)
            else:
                print(f"Модель для признака {feature} отстутствует")
                self.unprocessed_features.append(feature)

    def predict(self, new_data_row):
        """
        Делает предсказание для новой строки данных.
        """
        predictions = []
        feature_results = {}

        for feature in self.feature_names_list:
            # Получаем индекс из маппинга
            i = FEATURE_INDEX_MAPPING.get(feature)

            if i is None or (i, feature) not in self.models:
                feature_results[feature] = "Недоступно (нет модели)"
                continue

            # Формируем данные для одного признака
            new_data_for_model = pd.DataFrame({feature: new_data_row[feature]})

            # Преобразуем типы данных к числовым
            new_data_for_model = new_data_for_model.astype(float)

            # Масштабируем данные
            scaled_data = self.scalers[(i, feature)].transform(new_data_for_model[[feature]])
            pred = self.models[(i, feature)].predict(scaled_data)[0]

            # Сохраняем результат для признака
            feature_results[feature] = "Аномалия" if pred == -1 else "Нормальная точка"
            predictions.append(pred)

        # Преобразуем предсказания (-1 и 1) в бинарные метки (0 и 1)
        binary_predictions = [1 if p == -1 else 0 for p in predictions]

        # Проверяем количество аномальных признаков
        anomaly_count = sum(binary_predictions)  # Количество аномальных признаков
        final_prediction = "Аномалия" if anomaly_count >= ANOMALY_THRESHOLD else "Нормальная точка"

        return final_prediction, feature_results