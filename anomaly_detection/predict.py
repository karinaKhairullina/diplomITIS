import pandas as pd
from anomaly_detection.config import FEATURE_INDEX_MAPPING, ALTERNATIVE_NAMES
from anomaly_detection.cache import model_cache

STATUS_NO_MODEL = "Недоступно (нет модели)"
STATUS_ANOMALY = "Аномалия"
STATUS_NORMAL = "Нормальная точка"


class CombinedModel:
    def __init__(self, original_feature_names_list, feature_index_mapping, alternative_names, threshold=2):
        self.original_feature_names_list = [feature.strip() for feature in original_feature_names_list]
        self.normalized_feature_names_list = [
            alternative_names.get(feature, feature)
            for feature in self.original_feature_names_list
        ]
        self.models = model_cache.models
        self.scalers = model_cache.scalers
        self.feature_index_mapping = feature_index_mapping
        self.threshold = threshold

    def predict(self, data: pd.DataFrame):
        data_renamed = self.aggregate_columns(data)
        results = []

        for normalized_feature in self.normalized_feature_names_list:
            results.append(self.get_feature_status(normalized_feature, data_renamed))

        has_valid_predictions = any(
            any(status in [STATUS_ANOMALY, STATUS_NORMAL] for status in row) for row in results
        )
        if not has_valid_predictions:
            unavailable_features = [
                feature for feature, statuses in zip(self.original_feature_names_list, results)
                if all(status == STATUS_NO_MODEL for status in statuses)
            ]
            raise ValueError(
                "Отсутствуют модели или данные для признаков: " + ", ".join(unavailable_features)
            )

        results_by_row = list(zip(*results))
        final_results = []

        for row_statuses in results_by_row:
            anomaly_count = sum(1 for status in row_statuses if status == STATUS_ANOMALY)
            final_prediction = STATUS_ANOMALY if anomaly_count >= self.threshold else STATUS_NORMAL
            final_results.append((final_prediction, row_statuses))

        return final_results

    def get_feature_status(self, normalized_feature, data_renamed):
        i = self.feature_index_mapping.get(normalized_feature)
        key = (i, normalized_feature)

        if i is None or key not in self.models or normalized_feature not in data_renamed.columns:
            return [STATUS_NO_MODEL] * len(data_renamed)

        column_data = data_renamed[normalized_feature]

        if column_data.dropna().empty or not pd.to_numeric(column_data, errors='coerce').notna().all():
            return [STATUS_NO_MODEL] * len(column_data)

        scaled_data = self.scalers[key].transform(column_data.fillna(0).values.reshape(-1, 1))
        model = self.models[key]
        preds = model.predict(scaled_data)

        return [
            STATUS_ANOMALY if p == -1 else STATUS_NORMAL
            for p in preds
        ]

    def aggregate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        data_renamed = data.rename(columns=ALTERNATIVE_NAMES)

        aggregated = pd.DataFrame()
        for col in data_renamed.columns.unique():
            col_data = data_renamed.loc[:, data_renamed.columns == col]
            aggregated[col] = col_data.bfill(axis=1).iloc[:, 0]

        return aggregated