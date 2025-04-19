from django.shortcuts import render
import pandas as pd
from anomaly_detection.predict import CombinedModel, STATUS_ANOMALY, STATUS_NO_MODEL
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
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            else:
                raise ValueError("Поддерживаются только файлы формата CSV и JSON.")
        except EmptyDataError:
            return render(request, 'index.html', {'error': "Загруженный файл пустой или не содержит данных."})
        except Exception as e:
            return render(request, 'index.html', {'error': f"Ошибка при чтении файла: {str(e)}"})

        if df.empty:
            return render(request, 'index.html', {'error': "Загруженный файл пустой или не содержит данных."})

        original_feature_names = [f.strip() for f in df.columns.tolist()]
        combined_model = CombinedModel(original_feature_names, FEATURE_INDEX_MAPPING, ALTERNATIVE_NAMES)

        original_rows = df.copy()

        try:
            predictions = combined_model.predict(df)
        except Exception as e:
            return render(request, 'index.html', {'error': f"Ошибка при предсказании. {str(e)}"})

        for idx, (final_prediction, row_results) in enumerate(predictions):
            try:
                row_values = original_rows.iloc[idx]
            except IndexError:
                continue

            if row_values.isna().all():
                continue

            formatted_row = {
                'row_number': idx + 1,
                'values': [],
                'final_prediction': final_prediction
            }
            missing_features = []

            for original_feature, status in zip(original_feature_names, row_results):
                raw_value = row_values.get(original_feature, None)

                if status == STATUS_NO_MODEL:
                    display_value = raw_value if pd.notna(raw_value) else STATUS_NO_MODEL
                    formatted_row['values'].append({
                        'feature': original_feature,
                        'value': display_value,
                        'status': 'missing'
                    })
                    missing_features.append(original_feature)
                else:
                    formatted_row['values'].append({
                        'feature': original_feature,
                        'value': raw_value,
                        'status': status
                    })

            if final_prediction == STATUS_ANOMALY:
                results.append(formatted_row)

            missing_features_set.update(missing_features)

    elif request.method == 'POST':
        return render(request, 'index.html', {
            'error': "Файл не загружен. Пожалуйста, прикрепите CSV или JSON-файл для анализа."
        })

    return render(request, 'index.html', {
        'results': results,
        'missing_features_list': sorted(missing_features_set),
        'error': error
    })
