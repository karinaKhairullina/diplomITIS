import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from .forms import FileUploadForm
from .utils.analyzer import analyze_data

def index(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']

        # Загрузка CSV или JSON
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
            else:
                return JsonResponse({"error": "Поддерживаются только CSV и JSON файлы."}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Ошибка чтения файла: {str(e)}"}, status=400)

        # Прогоняем через модель
        preds, n_anomalies = analyze_data(df)
        df['anomaly'] = preds

        # Группируем по player_id и делаем стиль
        grouped_tables = []
        for player_id, group in df.groupby('player_id'):
            # Применяем стиль, а индекс убираем с помощью index=False
            styled = group.style.apply(
                lambda row: ['background-color: #f9d6d5' if row['anomaly'] == -1 else '' for _ in row],
                axis=1
            )
            html = styled.to_html(index=False)  # Убираем индекс
            grouped_tables.append((player_id, html))

        return render(request, 'index.html', {
            'grouped_tables': grouped_tables,
            'n_anomalies': n_anomalies,
            'file': file.name
        })

    # GET-запрос — форма
    form = FileUploadForm()
    return render(request, 'index.html', {'form': form})
