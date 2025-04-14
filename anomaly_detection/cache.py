import os
import joblib
from anomaly_detection.config import MODELS_DIR, FEATURE_INDEX_MAPPING

class ModelCache:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def load_models(self):
        """
        Загружает все модели и масштабировщики в память.
        """
        for feature in FEATURE_INDEX_MAPPING:
            i = FEATURE_INDEX_MAPPING.get(feature)
            if i is None:
                continue

            model_path = os.path.join(MODELS_DIR, f'model_{i}_{feature}.joblib')
            scaler_path = os.path.join(MODELS_DIR, f'scaler_{i}_{feature}.joblib')

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[(i, feature)] = joblib.load(model_path)
                    self.scalers[(i, feature)] = joblib.load(scaler_path)
                except Exception as e:
                    print(f"Ошибка при загрузке модели для признака {feature}: {str(e)}")

# Глобальный экземпляр кэша
model_cache = ModelCache()
model_cache.load_models()