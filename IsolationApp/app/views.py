# views.py

from django.shortcuts import render
import pandas as pd
from anomaly_detection.predict import CombinedModel
from anomaly_detection.config import MODELS_DIR

def index(request):
    results = []
    unprocessed_features_message = ""
    error = ""

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # Определяем формат файла
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

        # Получаем список признаков из загруженного файла
        feature_names_list = df.columns.tolist()  # Это плоский список

        # Создаем экземпляр модели
        combined_model = CombinedModel(MODELS_DIR, feature_names_list)

        # Анализируем данные
        for _, row in df.iterrows():
            new_data_row = pd.DataFrame([row])
            final_prediction, feature_results = combined_model.predict(new_data_row)

            # Формируем результат для вывода
            formatted_row = {
                'values': [],
            }

            for feature in feature_names_list:
                value = row.get(feature, "Нет данных")
                status = feature_results.get(feature, "Недоступно")
                formatted_row['values'].append({
                    'feature': feature,
                    'value': value,
                    'status': status
                })

            # Сохраняем только строки с аномалиями
            if final_prediction == "Аномалия":
                results.append(formatted_row)


    return render(request, 'index.html', {
        'results': results,
        'unprocessed_features_message': unprocessed_features_message,
        'error': error
    })