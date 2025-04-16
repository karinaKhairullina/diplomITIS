from anomaly_detection.config import FEATURE_INDEX_MAPPING, ALTERNATIVE_NAMES
from anomaly_detection.cache import model_cache


class CombinedModel:
    def __init__(self, original_feature_names_list, feature_index_mapping, alternative_names):
        # Оригинальные названия признаков
        self.original_feature_names_list = [feature.strip() for feature in original_feature_names_list]

        # Нормализованные названия признаков
        self.normalized_feature_names_list = [
            alternative_names.get(feature.strip(), feature.strip())
            for feature in original_feature_names_list
        ]

        self.unprocessed_features = []

        # Проверяем, есть ли модели в кэше
        self.models = model_cache.models
        self.scalers = model_cache.scalers

        print("Ожидаемые признаки на основе обучающей модели:")
        print(self.original_feature_names_list)
        print("Нормализованные признаки на основе обучающей модели:")
        print(self.normalized_feature_names_list)

        for normalized_feature in self.normalized_feature_names_list:
            i = feature_index_mapping.get(normalized_feature)
            if i is None or (i, normalized_feature) not in self.models:
                self.unprocessed_features.append(normalized_feature)
                print(f"Модель для признака {normalized_feature} отсутствует в кэше")

    def predict(self, data):
        results = []

        # Печать всех переданных признаков
        print("Признаки, передаваемые для предсказания:")
        print(data.columns.tolist())

        # Преобразуем переданные данные, чтобы нормализованные признаки совпали с теми, что использовались при обучении
        data_renamed = data.rename(columns=ALTERNATIVE_NAMES)

        print("Данные после переименования признаков в нормализованные:")
        print(data_renamed.columns.tolist())

        for original_feature, normalized_feature in zip(
                self.original_feature_names_list, self.normalized_feature_names_list
        ):
            print(f"Проверка признака: {original_feature} (нормализованный: {normalized_feature})")
            i = FEATURE_INDEX_MAPPING.get(normalized_feature)
            if i is None or (i, normalized_feature) not in self.models:
                results.append(["Недоступно (нет модели)"] * len(data))
                print(f"Модель для признака {normalized_feature} отсутствует.")
                continue

            # Масштабируем данные для текущего признака
            scaler = self.scalers[(i, normalized_feature)]
            try:
                # Используем нормализованное название для извлечения данных после переименования
                scaled_data = scaler.transform(data_renamed[[normalized_feature]].astype(float))
            except KeyError:
                results.append(["Недоступно (нет данных)"] * len(data))
                print(f"Признак {original_feature} отсутствует в данных.")
                continue

            # Делаем предсказания для всех строк
            model = self.models[(i, normalized_feature)]
            preds = model.predict(scaled_data)

            # Преобразуем предсказания в текстовый формат
            feature_results = ["Аномалия" if p == -1 else "Нормальная точка" for p in preds]
            results.append(feature_results)

        # Транспонируем результаты
        results = list(zip(*results))

        # Формируем финальные результаты
        final_results = []
        for row_results in results:
            anomaly_count = sum(1 for r in row_results if r == "Аномалия")
            final_prediction = "Аномалия" if anomaly_count >= 2 else "Нормальная точка"
            final_results.append((final_prediction, row_results))

        return final_results
