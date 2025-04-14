from django.shortcuts import render
import pandas as pd
from anomaly_detection.predict import CombinedModel

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

        feature_names_list = df.columns.tolist()
        combined_model = CombinedModel(feature_names_list)
        predictions = combined_model.predict(df)

        for idx, (final_prediction, row_results) in enumerate(predictions):
            formatted_row = {
                'values': []
            }
            missing_features = []

            for feature, status in zip(feature_names_list, row_results):
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

            if final_prediction == "Аномалия":
                results.append(formatted_row)

            missing_features_set.update(missing_features)

    return render(request, 'index.html', {
        'results': results,
        'missing_features_list': sorted(missing_features_set),
        'error': error
    })
