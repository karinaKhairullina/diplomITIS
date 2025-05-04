import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import umap


def calculate_dynamic_threshold(scores, percentile=1):
    """
    Вычисляет динамический порог: например, 1% самых низких значений score_samples.
    """
    return np.percentile(scores, percentile)

def plot_anomaly_score_histogram(scores, threshold=None, save_path='score_histogram.png'):
    """
    Строит гистограмму anomaly_score с KDE и линией порога.
    """
    plt.figure(figsize=(10, 6))

    # Гистограмма
    sns.histplot(scores, bins=50, kde=True, color='blue', alpha=0.6, stat='density', label='Гистограмма')

    # Линия порога, если задана
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Порог: {threshold:.3f}')

    plt.title('Распределение anomaly_score')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Плотность')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_umap_with_anomaly_scores(X_scaled, scores, save_path='media/umap_anomalies.png'):
    """
    Строит UMAP-график с цветовой градацией по anomaly_score.
    """
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    df_plot = pd.DataFrame(embedding, columns=['x', 'y'])
    df_plot['anomaly_score'] = scores

    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=df_plot,
        x='x',
        y='y',
        hue='anomaly_score',
        palette='coolwarm_r',
        alpha=0.8,
        s=50,
        edgecolor=None
    )
    plt.title('UMAP по Anomaly Score')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.legend(title='Anomaly Score', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

