from django.shortcuts import render
import pandas as pd
from anomaly_detection.predict import CombinedModel
from anomaly_detection.config import ALTERNATIVE_NAMES

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
                error = 'Неподдерживаемый формат файла'
                return render(request, 'index.html', {'error': error})
        except Exception as e:
            error = f'Ошибка при чтении файла: {str(e)}'
            return render(request, 'index.html', {'error': error})

        # Нормализуем названия признаков
        feature_names_list = [
            ALTERNATIVE_NAMES.get(feature.strip(), feature.strip())
            for feature in df.columns.tolist()
        ]

        combined_model = CombinedModel(feature_names_list)
        predictions = combined_model.predict(df)

        # Здесь predictions — это список списков с результатами для каждой строки
        for idx, (final_prediction, row_results) in enumerate(predictions):  # Изменение здесь
            formatted_row = {
                'values': []
            }
            missing_features = []

            for feature, status in zip(combined_model.feature_names_list, row_results):
                value = df.iloc[idx].get(feature, "Недоступно (нет модели)")
                if status == "Недоступно (нет модели)":
                    missing_features.append(feature)
                    continue
                else:
                    formatted_row['values'].append({
                        'feature': feature,
                        'value': value,
                        'status': status
                    })

            # Если хотя бы два признака аномальны, считаем всю строку аномальной
            anomaly_count = sum(1 for r in row_results if r == "Аномалия")
            final_prediction = "Аномалия" if anomaly_count >= 2 else "Нормальная точка"

            if final_prediction == "Аномалия":
                results.append(formatted_row)

            missing_features_set.update(missing_features)

    return render(request, 'index.html', {
        'results': results,
        'missing_features_list': sorted(missing_features_set),
        'error': error
    })
