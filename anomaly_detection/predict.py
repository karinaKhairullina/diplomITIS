from anomaly_detection.config import FEATURE_INDEX_MAPPING, ALTERNATIVE_NAMES
from anomaly_detection.cache import model_cache

class CombinedModel:
    def __init__(self, feature_names_list):
        self.feature_names_list = []
        self.unprocessed_features = []

        # Нормализуем названия признаков
        for feature in feature_names_list:
            normalized_feature = ALTERNATIVE_NAMES.get(feature.strip(), feature.strip())
            self.feature_names_list.append(normalized_feature)

            # Проверяем, есть ли модель для нормализованного признака
            i = FEATURE_INDEX_MAPPING.get(normalized_feature)
            if i is None or (i, normalized_feature) not in model_cache.models:
                self.unprocessed_features.append(feature)
                print(f"Модель для признака {feature} отсутствует в кэше")

    def predict(self, data):
        """
        Делает предсказание для всего набора данных.
        :param data: Полный DataFrame с данными
        :return: Список результатов для каждой строки
        """
        results = []
        for feature in self.feature_names_list:
            i = FEATURE_INDEX_MAPPING.get(feature)
            if i is None or (i, feature) not in model_cache.models:
                # Если модель недоступна, добавляем "Недоступно" для всех строк
                results.append(["Недоступно (нет модели)"] * len(data))
                continue

            # Масштабируем данные для текущего признака
            scaler = model_cache.scalers[(i, feature)]
            scaled_data = scaler.transform(data[[feature]].astype(float))

            # Делаем предсказания для всех строк
            model = model_cache.models[(i, feature)]
            preds = model.predict(scaled_data)

            # Преобразуем предсказания в текстовый формат
            feature_results = ["Аномалия" if p == -1 else "Нормальная точка" for p in preds]
            results.append(feature_results)

        # Транспонируем результаты для удобства работы
        results = list(zip(*results))

        # Формируем финальные результаты
        final_results = []
        for row_results in results:
            # Подсчитываем количество аномалий в строке
            anomaly_count = sum(1 for r in row_results if r == "Аномалия")
            final_prediction = "Аномалия" if anomaly_count >= 2 else "Нормальная точка"
            final_results.append((final_prediction, row_results))

        # Для отладки
        print("Предсказания по строкам:")
        for i, (fp, rs) in enumerate(final_results):
            print(f"Строка {i}: {fp} → {rs}")

        return final_results
