from anomaly_detection.config import FEATURE_INDEX_MAPPING, ANOMALY_THRESHOLD
from anomaly_detection.cache import model_cache

class CombinedModel:
    def __init__(self, feature_names_list):
        self.feature_names_list = [feature.strip() for feature in feature_names_list]
        self.unprocessed_features = []

        # Проверяем, есть ли модели в кэше
        self.models = model_cache.models
        self.scalers = model_cache.scalers

        for feature in self.feature_names_list:
            i = FEATURE_INDEX_MAPPING.get(feature)

            if i is None or (i, feature) not in self.models:
                self.unprocessed_features.append(feature)
                print(f"Модель для признака {feature} отсутствует в кэше")
                continue

    def predict(self, data):
        """
        Делает предсказание для всего набора данных.
        :param data: Полный DataFrame с данными
        :return: Список результатов для каждой строки
        """
        results = []

        for feature in self.feature_names_list:
            i = FEATURE_INDEX_MAPPING.get(feature)

            if i is None or (i, feature) not in self.models:
                # Если модель недоступна, добавляем "Недоступно" для всех строк
                results.append(["Недоступно (нет модели)"] * len(data))
                continue

            # Масштабируем данные для текущего признака
            scaler = self.scalers[(i, feature)]
            scaled_data = scaler.transform(data[[feature]].astype(float))

            # Делаем предсказания для всех строк
            model = self.models[(i, feature)]
            preds = model.predict(scaled_data)

            # Преобразуем предсказания в текстовый формат
            feature_results = ["Аномалия" if p == -1 else "Нормальная точка" for p in preds]
            results.append(feature_results)

        # Транспонируем результаты для удобства работы
        results = list(zip(*results))

        # Формируем финальные результаты
        final_results = []
        for row_results in results:
            binary_predictions = [1 if r == "Аномалия" else 0 for r in row_results]
            anomaly_count = sum(binary_predictions)
            final_prediction = "Аномалия" if anomaly_count >= ANOMALY_THRESHOLD else "Нормальная точка"
            final_results.append((final_prediction, row_results))

        return final_results