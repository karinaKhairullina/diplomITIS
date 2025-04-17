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

        # Модели и масштабаторы из кэша
        self.models = model_cache.models
        self.scalers = model_cache.scalers

        for normalized_feature in self.normalized_feature_names_list:
            i = feature_index_mapping.get(normalized_feature)
            if i is None or (i, normalized_feature) not in self.models:
                self.unprocessed_features.append(normalized_feature)

    def predict(self, data):
        # Проверка на корректность входных данных
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Ожидается таблица формата pandas.DataFrame.")

        if data.empty:
            raise ValueError("Загруженный файл пуст или не содержит данных.")

        results = []

        # Агрегация данных: объединяем значения всех альтернативных признаков
        data_renamed = self.aggregate_columns(data)

        for original_feature, normalized_feature in zip(
                self.original_feature_names_list, self.normalized_feature_names_list
        ):
            i = FEATURE_INDEX_MAPPING.get(normalized_feature)
            if i is None or (i, normalized_feature) not in self.models:
                results.append(["Недоступно (нет модели)"] * len(data_renamed))
                continue

            # Проверяем наличие признака в агрегированных данных
            if normalized_feature not in data_renamed.columns:
                results.append(["Недоступно (нет данных)"] * len(data_renamed))
                continue

            # Получаем и очищаем столбец
            column_data = data_renamed[normalized_feature].dropna()
            if column_data.empty:
                results.append(["Недоступно (нет данных)"] * len(data_renamed))
                continue

            # Убедимся, что column_data является одномерным
            if column_data.ndim > 1:
                column_data = column_data.squeeze()

            # Масштабируем значения (приводим к формату [ [x], [y], ... ])
            scaled_data = self.scalers[(i, normalized_feature)].transform(column_data.values.reshape(-1, 1))

            # Получаем предсказания модели
            model = self.models[(i, normalized_feature)]
            preds = model.predict(scaled_data)

            # Конвертируем предсказания в текст
            feature_results = ["Аномалия" if p == -1 else "Нормальная точка" for p in preds]

            # Восстанавливаем длину столбца, заполняя NaN места, если нужно
            full_results = []
            pred_index = 0
            for val in data_renamed[normalized_feature]:
                if pd.isna(val):
                    full_results.append("Недоступно (NaN)")
                else:
                    full_results.append(feature_results[pred_index])
                    pred_index += 1

            results.append(full_results)

        # Транспонируем, чтобы получить предсказания по строкам
        results = list(zip(*results))

        # Финальный вывод: если 2 или более аномалий — метим всю строку как аномальную
        final_results = []
        for row_results in results:
            anomaly_count = sum(1 for r in row_results if r == "Аномалия")
            final_prediction = "Аномалия" if anomaly_count >= 2 else "Нормальная точка"
            final_results.append((final_prediction, row_results))

        return final_results

    def aggregate_columns(self, data):
        """Объединяем все значения альтернативных признаков в один столбец на каждый нормализованный признак."""

        # Удаляем столбцы с ненужными именами, такими как "Unnamed: 0" или "Unnamed: 1"
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        # Переименовываем столбцы согласно ALTERNATIVE_NAMES
        data_renamed = data.rename(columns=ALTERNATIVE_NAMES)

        # Логируем все переименованные столбцы
        print("Переименованные столбцы:", data_renamed.columns.tolist())

        aggregated = {}

        # Обрабатываем каждый столбец для агрегации
        for col in set(data_renamed.columns):
            # Собираем все столбцы с одинаковым нормализованным именем
            cols_with_same_name = [c for c in data_renamed.columns if c == col]

            # Логируем, какие столбцы объединяются
            print(f"Объединяем столбцы для признака {col}: {cols_with_same_name}")

            # Объединяем значения из этих столбцов в один Series
            combined_values = pd.concat(
                [data_renamed[c].dropna() for c in cols_with_same_name],
                ignore_index=True
            )

            # Убедимся, что combined_values является одномерным массивом
            if isinstance(combined_values, pd.DataFrame):
                combined_values = combined_values.stack().reset_index(drop=True)

            aggregated[col] = combined_values

        result_df = pd.DataFrame(aggregated)

        return result_df


    def clean_data(self, data):
        """Очистка данных: замена запятых на точки, приведение к числам, NaN вместо ошибок."""
        data = data.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
        data = data.apply(pd.to_numeric, errors='coerce')
        return data
