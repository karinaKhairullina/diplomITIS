from django.shortcuts import render
from django.http import JsonResponse
from .forms import FileUploadForm
import pandas as pd
from io import StringIO
from .utils.preprocess import map_feature_aliases
from .utils.analyzer import model, scaler, prepare_test_data, REQUIRED_FEATURES
from .utils.visualization import plot_anomaly_score_histogram, calculate_dynamic_threshold,plot_umap_with_anomaly_scores

def group_by_player_and_convert_to_html(df):
    grouped = {}
    for player_id, group in df.groupby('player_id'):
        grouped[str(player_id)] = group.to_html(index=False)
    return grouped

def index(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        try:
            file_content = file.read().decode('utf-8')
            if file.name.endswith('.csv'):
                data_stream = StringIO(file_content)
                chunks = pd.read_csv(data_stream, chunksize=5000)
            else:
                return JsonResponse({"error": "Поддерживаются только CSV файлы."}, status=400)

            anomalies = []
            all_data = []
            all_scaled = []
            all_scores = []

            for chunk in chunks:
                chunk = map_feature_aliases(chunk)
                chunk_prepared = prepare_test_data(chunk, required_columns=REQUIRED_FEATURES)
                X_scaled = scaler.transform(chunk_prepared)

                # Используем метод score_samples для расчета anomaly score
                scores = model.score_samples(X_scaled)  # anomaly_score
                all_scores.extend(scores)
                all_scaled.append(X_scaled)
                chunk['anomaly_score'] = scores

                all_data.append(chunk)

            # Рассчитываем порог по 1-му процентилю
            threshold = calculate_dynamic_threshold(all_scores, percentile=1)

            # Повторно присваиваем флаг аномалий на основе порога
            all_data_df = pd.concat(all_data)
            all_data_df['Аномалия'] = ['Да' if s < threshold else 'Нет' for s in all_data_df['anomaly_score']]
            all_anomalies_df = all_data_df[all_data_df['Аномалия'] == 'Да']
            all_scaled_array = pd.concat([pd.DataFrame(arr) for arr in all_scaled])

            # Сохраняем график гистограммы
            plot_anomaly_score_histogram(all_scores, threshold=threshold, save_path='media/score_histogram.png')
            plot_umap_with_anomaly_scores(
                X_scaled=all_scaled_array.values,
                scores=all_scores,
                save_path='media/umap_anomalies.png'
            )

            return JsonResponse({
                'status': 'completed',
                'n_anomalies': len(all_anomalies_df),
                'anomalies': group_by_player_and_convert_to_html(all_anomalies_df),
                'all_rows': group_by_player_and_convert_to_html(all_data_df)
            }, status=202)

        except Exception as e:
            return JsonResponse({"error": f"Ошибка при обработке файла: {str(e)}"}, status=400)

    return render(request, 'index.html', {'form': FileUploadForm()})
