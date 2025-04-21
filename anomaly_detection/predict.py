import pandas as pd
from anomaly_detection.config import FEATURE_INDEX_MAPPING, ALTERNATIVE_NAMES
from anomaly_detection.cache import model_cache

STATUS_NO_MODEL = "Недоступно (нет модели)"
STATUS_NO_DATA = "Недоступно (нет данных)"
STATUS_ANOMALY = "Аномалия"
STATUS_NORMAL = "Нормальная точка"

class CombinedModel:
    def __init__(self, original_feature_names_list, feature_index_mapping, alternative_names, threshold=2):
        self.original_feature_names_list = [feature.strip() for feature in original_feature_names_list]
        self.normalized_feature_names_list = [
            alternative_names.get(feature, feature) for feature in self.original_feature_names_list
        ]
        self.models = model_cache.models
        self.scalers = model_cache.scalers
        self.feature_index_mapping = feature_index_mapping
        self.threshold = threshold

    def aggregate_columns(self, data: pd.DataFrame):
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        # Переименовываем столбцы согласно альтернативам
        data_renamed = data.rename(columns=ALTERNATIVE_NAMES)

        aggregated = pd.DataFrame()
        feature_origin_map = {}
        index_map = {}

        # Агрегируем данные
        for normalized_feature in self.normalized_feature_names_list:
            original_features = [col for col in data.columns if ALTERNATIVE_NAMES.get(col, col) == normalized_feature]
            feature_origin_map[normalized_feature] = original_features

            # Сохраняем индексы для каждого исходного признака
            index_map[normalized_feature] = {}

            for original_feature in original_features:
                if original_feature in data.columns:
                    index_map[normalized_feature][original_feature] = data[original_feature].index.tolist()

            # Объединяем значения всех исходных признаков в один массив
            all_values = []
            for original_feature in original_features:
                if original_feature in data.columns:
                    all_values.extend(data[original_feature].dropna().tolist())

            if all_values:
                aggregated[normalized_feature] = pd.Series(all_values).reset_index(drop=True)

        return aggregated, feature_origin_map, index_map

    def get_feature_status(self, normalized_feature, values, positions):
        i = self.feature_index_mapping.get(normalized_feature)
        key = (i, normalized_feature)

        if i is None or key not in self.models:
            return {(idx, col): STATUS_NO_MODEL for idx, col in positions}

        df = pd.DataFrame({normalized_feature: values}).fillna(0)
        scaled_data = self.scalers[key].transform(df)
        preds = self.models[key].predict(scaled_data)

        # Вернем статус для каждого значения
        return {
            (idx, col): (STATUS_ANOMALY if pred == -1 else STATUS_NORMAL)
            for (idx, col), pred in zip(positions, preds)
        }

    def predict(self, data: pd.DataFrame):
        feature_status_flat = {}

        # Для каждого нормализованного признака
        for norm_feature in self.normalized_feature_names_list:
            original_cols = [col for col in data.columns if ALTERNATIVE_NAMES.get(col, col) == norm_feature]
            values = []
            positions = []

            # Собираем все значения и их позиции
            for col in original_cols:
                for idx, val in data[col].items():
                    if pd.notna(val):  # Убираем NaN значения
                        values.append(val)
                        positions.append((idx, col))

            if not values:
                continue

            preds = self.get_feature_status(norm_feature, values, positions)
            feature_status_flat.update(preds)

        # Аннотируем данные для каждого значения
        annotated_data = pd.DataFrame(index=data.index, columns=data.columns)
        for idx in data.index:
            for col in data.columns:
                val = data.at[idx, col]
                status = feature_status_flat.get((idx, col), STATUS_NO_DATA)
                if pd.notna(val):
                    annotated_data.at[idx, col] = f"{val} ({status})"
                else:
                    annotated_data.at[idx, col] = status


        # Подсчитываем количество аномалий в каждой строке
        row_results = []
        for idx in data.index:
            anomaly_count = sum(
                1 for col in data.columns
                if feature_status_flat.get((idx, col), STATUS_NO_DATA) == STATUS_ANOMALY
            )

            # Если количество аномалий больше или равно порогу, считаем строку аномальной
            if anomaly_count >= self.threshold:
                row_results.append(STATUS_ANOMALY)
            else:
                row_results.append(STATUS_NORMAL)

        return row_results, annotated_data
