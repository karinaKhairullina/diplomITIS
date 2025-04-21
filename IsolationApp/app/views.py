from django.shortcuts import render
import pandas as pd
from anomaly_detection.predict import CombinedModel, STATUS_ANOMALY, STATUS_NO_MODEL, STATUS_NO_DATA, STATUS_NORMAL
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
            # Загружаем файл в зависимости от его формата
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

        # Проверка на пустоту данных
        if df.empty:
            return render(request, 'index.html', {'error': "Загруженный файл пустой или не содержит данных."})

        # Проверка на корректность данных (некорректные буквы в числовых столбцах)
        invalid_columns = []
        for column in df.columns:
            if df[column].apply(lambda x: not isinstance(x, (int, float)) and not pd.isna(x)).any():
                invalid_columns.append(column)

        if invalid_columns:
            return render(request, 'index.html', {
                'error': f"Ошибка при предсказании. Отсутствуют модели или корректные данные для признаков: {', '.join(invalid_columns)}"
            })

        # Инициализация модели и выполнение предсказания
        original_feature_names = [f.strip() for f in df.columns.tolist()]
        combined_model = CombinedModel(original_feature_names, FEATURE_INDEX_MAPPING, ALTERNATIVE_NAMES)

        original_rows = df.copy()

        try:
            final_results, per_feature_results = combined_model.predict(df)
        except Exception as e:
            return render(request, 'index.html', {'error': f"Ошибка при предсказании. {str(e)}"})

        # Форматируем результат для отображения
        for idx, final_prediction in enumerate(final_results):
            try:
                row_values = original_rows.iloc[idx]
            except IndexError:
                continue

            if row_values.isna().all():
                continue

            formatted_row = {
                'row_number': original_rows.index[idx] + 1,
                'values': [],
                'final_prediction': final_prediction
            }
            missing_features = []

            for original_feature in original_feature_names:
                raw_value = row_values.get(original_feature, None)
                raw_status = per_feature_results.at[idx, original_feature]

                # Обработка статуса и значения
                if isinstance(raw_status, str) and '(' in raw_status and ')' in raw_status:
                    status = raw_status.split('(')[-1].rstrip(')')
                    value_part = raw_status.split('(')[0].strip()
                else:
                    status = raw_status
                    value_part = raw_status

                if status == STATUS_NO_MODEL:
                    status_class = 'missing'
                elif status == STATUS_NO_DATA or pd.isna(raw_value):
                    status_class = 'missing'
                else:
                    status_class = 'anomaly' if status == STATUS_ANOMALY else 'normal'

                formatted_row['values'].append({
                    'feature': original_feature,
                    'value': value_part,
                    'status': status_class
                })

                if status_class == 'missing':
                    missing_features.append(original_feature)

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

