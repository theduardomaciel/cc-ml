"""
Módulo de Otimização do Número de Clusters
Utiliza múltiplas métricas para determinar o K ótimo
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class KOptimizer:
    """
    Classe para otimização do número de clusters usando múltiplas métricas.
    """

    def __init__(self, k_range=(2, 11), random_state=42):
        """
        Inicializa o otimizador.

        Args:
            k_range (tuple): Range de valores de K a testar (min, max)
            random_state (int): Semente para reprodutibilidade
        """
        self.k_range = range(k_range[0], k_range[1])
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.metrics_history = []

    def optimize(self, data):
        """
        Testa diferentes valores de K e calcula métricas.

        Args:
            data (pd.DataFrame): Dados para clusterização

        Returns:
            pd.DataFrame: DataFrame com métricas para cada K
        """
        print("=" * 60)
        print("OTIMIZAÇÃO DO NÚMERO DE CLUSTERS")
        print("=" * 60)
        print(f"Testando K de {min(self.k_range)} a {max(self.k_range) - 1}...")
        print()

        # Normalizar dados
        data_normalized = self.scaler.fit_transform(data)

        # Testar cada valor de K
        for k in self.k_range:
            print(f"Testando K = {k}...", end=" ")

            kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = kmeans.fit_predict(data_normalized)

            metrics = {
                "k": k,
                "inertia": kmeans.inertia_,
                "silhouette": silhouette_score(data_normalized, labels),
                "calinski_harabasz": calinski_harabasz_score(data_normalized, labels),
                "davies_bouldin": davies_bouldin_score(data_normalized, labels),
            }

            self.metrics_history.append(metrics)
            print(f"✓ (Silhouette: {metrics['silhouette']:.3f})")

        df_metrics = pd.DataFrame(self.metrics_history)

        # Determinar K ótimo para cada métrica
        print("\n" + "=" * 60)
        print("RECOMENDAÇÕES POR MÉTRICA")
        print("=" * 60)

        k_elbow = self._find_elbow(df_metrics["k"].values, df_metrics["inertia"].values)
        k_silhouette = df_metrics.loc[df_metrics["silhouette"].idxmax(), "k"]
        k_calinski = df_metrics.loc[df_metrics["calinski_harabasz"].idxmax(), "k"]
        k_davies = df_metrics.loc[df_metrics["davies_bouldin"].idxmin(), "k"]

        print(f"✓ Método do Cotovelo (Elbow): K = {k_elbow}")
        print(f"✓ Silhouette Score (max): K = {int(k_silhouette)}")
        print(f"✓ Calinski-Harabasz (max): K = {int(k_calinski)}")
        print(f"✓ Davies-Bouldin (min): K = {int(k_davies)}")

        # Votação para K ótimo
        votes = [k_elbow, k_silhouette, k_calinski, k_davies]
        k_optimal = int(pd.Series(votes).mode()[0])

        print(f"\n🏆 K ÓTIMO RECOMENDADO: {k_optimal}")
        print(f"   (Baseado em votação das 4 métricas)")

        return df_metrics, k_optimal

    def _find_elbow(self, k_values, inertias):
        """
        Encontra o ponto de cotovelo usando o método da segunda derivada.

        Args:
            k_values (np.ndarray): Valores de K testados
            inertias (np.ndarray): Valores de inércia correspondentes

        Returns:
            int: K no ponto de cotovelo
        """
        # Normalizar inércia para 0-1
        inertias_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())
        k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())

        # Calcular distância de cada ponto à linha que conecta primeiro e último ponto
        line_vec = np.array(
            [k_norm[-1] - k_norm[0], inertias_norm[-1] - inertias_norm[0]]
        )
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

        distances = []
        for i in range(len(k_values)):
            point_vec = np.array(
                [k_norm[i] - k_norm[0], inertias_norm[i] - inertias_norm[0]]
            )
            dist = np.abs(np.cross(line_vec_norm, point_vec))
            distances.append(dist)

        elbow_idx = np.argmax(distances)
        return int(k_values[elbow_idx])

    def plot_optimization(self, df_metrics, save_path="results/k_optimization.png"):
        """
        Plota gráficos de otimização para todas as métricas.

        Args:
            df_metrics (pd.DataFrame): DataFrame com métricas
            save_path (str): Caminho para salvar o gráfico
        """
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Otimização do Número de Clusters (K)", fontsize=16, fontweight="bold"
        )

        # 1. Método do Cotovelo
        ax1 = axes[0, 0]
        ax1.plot(
            df_metrics["k"], df_metrics["inertia"], "o-", linewidth=2, markersize=8
        )
        k_elbow = self._find_elbow(df_metrics["k"].values, df_metrics["inertia"].values)
        ax1.axvline(
            x=k_elbow,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"K ótimo = {k_elbow}",
        )
        ax1.set_xlabel("Número de Clusters (K)", fontsize=11)
        ax1.set_ylabel("Inércia", fontsize=11)
        ax1.set_title(
            "Método do Cotovelo (Elbow Method)", fontsize=12, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Silhouette Score
        ax2 = axes[0, 1]
        ax2.plot(
            df_metrics["k"],
            df_metrics["silhouette"],
            "o-",
            color="green",
            linewidth=2,
            markersize=8,
        )
        k_best = df_metrics.loc[df_metrics["silhouette"].idxmax(), "k"]
        ax2.axvline(
            x=k_best,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"K ótimo = {int(k_best)}",
        )
        ax2.set_xlabel("Número de Clusters (K)", fontsize=11)
        ax2.set_ylabel("Silhouette Score", fontsize=11)
        ax2.set_title(
            "Silhouette Score (Maior é Melhor)", fontsize=12, fontweight="bold"
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Calinski-Harabasz Score
        ax3 = axes[1, 0]
        ax3.plot(
            df_metrics["k"],
            df_metrics["calinski_harabasz"],
            "o-",
            color="orange",
            linewidth=2,
            markersize=8,
        )
        k_best = df_metrics.loc[df_metrics["calinski_harabasz"].idxmax(), "k"]
        ax3.axvline(
            x=k_best,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"K ótimo = {int(k_best)}",
        )
        ax3.set_xlabel("Número de Clusters (K)", fontsize=11)
        ax3.set_ylabel("Calinski-Harabasz Score", fontsize=11)
        ax3.set_title(
            "Calinski-Harabasz Score (Maior é Melhor)", fontsize=12, fontweight="bold"
        )
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Davies-Bouldin Score
        ax4 = axes[1, 1]
        ax4.plot(
            df_metrics["k"],
            df_metrics["davies_bouldin"],
            "o-",
            color="purple",
            linewidth=2,
            markersize=8,
        )
        k_best = df_metrics.loc[df_metrics["davies_bouldin"].idxmin(), "k"]
        ax4.axvline(
            x=k_best,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"K ótimo = {int(k_best)}",
        )
        ax4.set_xlabel("Número de Clusters (K)", fontsize=11)
        ax4.set_ylabel("Davies-Bouldin Score", fontsize=11)
        ax4.set_title(
            "Davies-Bouldin Score (Menor é Melhor)", fontsize=12, fontweight="bold"
        )
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Gráfico de otimização salvo em: {save_path}")
        plt.close()

    def save_metrics(self, df_metrics, save_path="results/k_optimization_metrics.csv"):
        """
        Salva as métricas de otimização em arquivo CSV.

        Args:
            df_metrics (pd.DataFrame): DataFrame com métricas
            save_path (str): Caminho para salvar o arquivo
        """
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        df_metrics.to_csv(save_path, index=False)
        print(f"✓ Métricas de otimização salvas em: {save_path}")


if __name__ == "__main__":
    # Exemplo de uso
    from kmeans_clustering import KMeansEpithelialClusterer

    # Carregar dados
    clusterer = KMeansEpithelialClusterer()
    df = clusterer.load_and_select_data("data/RTVue_20221110_MLClass.csv")
    df_clean = clusterer.preprocess_data(df)

    # Otimizar K
    optimizer = KOptimizer()
    df_metrics, k_optimal = optimizer.optimize(df_clean)
    optimizer.plot_optimization(df_metrics)
    optimizer.save_metrics(df_metrics)
