from django.shortcuts import render
import pandas as pd
from anomaly_detection.predict import CombinedModel
from anomaly_detection.config import FEATURE_INDEX_MAPPING, ALTERNATIVE_NAMES
from pandas.errors import EmptyDataError


def index(request):
    results = []
    error = ""
    missing_features_set = set()

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_extension = uploaded_file.name.split('.')[-1].lower()

        try:
            # Чтение файла в зависимости от расширения
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            else:
                error = 'Неподдерживаемый формат файла'
                return render(request, 'index.html', {'error': error})

        except EmptyDataError:
            error = "Загруженный файл пустой или не содержит данных."
            return render(request, 'index.html', {'error': error})

        except Exception:
            error = 'Ошибка при чтении файла'
            return render(request, 'index.html', {'error': error})

        if df.empty:
            error = "Загруженный файл пустой или не содержит данных."
            return render(request, 'index.html', {'error': error})

        # Оригинальные названия признаков
        original_feature_names_list = [feature.strip() for feature in df.columns.tolist()]

        # Инициализация модели
        combined_model = CombinedModel(original_feature_names_list, FEATURE_INDEX_MAPPING, ALTERNATIVE_NAMES)

        # Очистка данных
        df = combined_model.clean_data(df)
        original_rows = df.copy()  # Сохраняем копию до агрегации

        try:
            predictions = combined_model.predict(df)
        except Exception:
            error = "Ошибка при предсказании. Проверьте содержимое файла."
            return render(request, 'index.html', {'error': error})

        # Обработка предсказаний
        for idx, (final_prediction, row_results) in enumerate(predictions):
            try:
                row_values = original_rows.iloc[idx]
            except IndexError:
                continue  # Пропустить, если вдруг индекс вышел за пределы

            if row_values.isna().all():
                continue

            formatted_row = {
                'row_number': idx + 1,
                'values': []
            }
            missing_features = []

            for original_feature, status in zip(original_feature_names_list, row_results):
                value = row_values.get(original_feature, "Недоступно (нет модели)")

                if status in ["Недоступно (нет модели)", "Недоступно (нет данных)", "Недоступно (NaN)"]:
                    formatted_row['values'].append({
                        'feature': original_feature,
                        'value': value,
                        'status': 'missing'
                    })
                    missing_features.append(original_feature)
                else:
                    formatted_row['values'].append({
                        'feature': original_feature,
                        'value': value,
                        'status': status
                    })

            anomaly_count = sum(1 for r in row_results if r == "Аномалия")
            final_prediction = "Аномалия" if anomaly_count >= 2 else "Нормальная точка"

            if final_prediction == "Аномалия":
                formatted_row['final_prediction'] = final_prediction
                results.append(formatted_row)

            missing_features_set.update(missing_features)

    return render(request, 'index.html', {
        'results': results,
        'missing_features_list': sorted(missing_features_set),
        'error': error
    })
