import pandas as pd
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

        for normalized_feature in self.normalized_feature_names_list:
            i = feature_index_mapping.get(normalized_feature)
            if i is None or (i, normalized_feature) not in self.models:
                self.unprocessed_features.append(normalized_feature)

    def predict(self, data):
        results = []

        # Преобразуем переданные данные, чтобы нормализованные признаки совпали с теми, что использовались при обучении
        data_renamed = self.aggregate_columns(data)

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

    def aggregate_columns(self, data):
        """Агрегируем все колонки с одинаковыми нормализованными названиями (например, Elim, KB => Kills)."""
        # Шаг 1: Переименуем колонки
        data_renamed = data.rename(columns=ALTERNATIVE_NAMES)

        # Шаг 2: Собираем, какие колонки теперь имеют одинаковое имя
        new_columns = data_renamed.columns
        aggregated = pd.DataFrame()

        for col in set(new_columns):
            cols_with_same_name = [c for c in new_columns if c == col]

            if len(cols_with_same_name) == 1:
                # Только одна колонка — оставляем как есть
                aggregated[col] = data_renamed[col]
            else:
                # Несколько колонок — агрегируем (например, берём среднее по строке)
                aggregated[col] = data_renamed[cols_with_same_name].mean(axis=1, skipna=True)

        return aggregated

    def clean_data(self, data):
        """Функция для обработки данных: замена запятых на точки, преобразование строк в числа, обработка NaN."""
        # Заменяем запятые на точки в строковых данных
        data = data.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

        # Преобразуем строки в числовой формат, ошибки преобразования заменяются на NaN
        data = data.apply(pd.to_numeric, errors='coerce')

        return data
