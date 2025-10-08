import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from kmeans_clustering import KMeansEpithelialClusterer
import warnings

warnings.filterwarnings("ignore")


class KOptimizer:
    """
    Classe para otimizar o número de clusters (k) para análise de espessura epitelial
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.clusterer = KMeansEpithelialClusterer()

    def analyze_optimal_k(self, max_k=10):
        """
        Analisa o número ótimo de clusters usando múltiplas métricas
        """
        print("=" * 70)
        print("OTIMIZAÇÃO DO NÚMERO DE CLUSTERS (K)")
        print("=" * 70)

        # Carrega e preprocessa dados
        self.clusterer.load_data(self.data_path)
        self.clusterer.preprocess_data()

        # Encontra número ótimo de clusters
        metrics = self.clusterer.find_optimal_clusters(max_clusters=max_k)
        metrics_df = pd.DataFrame(metrics)

        # Recomendações baseadas em diferentes critérios
        print("\n" + "=" * 70)
        print("RECOMENDAÇÕES DE K ÓTIMO")
        print("=" * 70)

        # Melhor Silhouette
        best_silhouette_idx = metrics_df["silhouette"].idxmax()
        k_silhouette = int(metrics_df.loc[best_silhouette_idx, "n_clusters"])
        print(f"\n📊 Melhor Silhouette Score: k = {k_silhouette}")
        print(f"   Score: {metrics_df.loc[best_silhouette_idx, 'silhouette']:.4f}")

        # Melhor Calinski-Harabasz
        best_ch_idx = metrics_df["calinski_harabasz"].idxmax()
        k_ch = int(metrics_df.loc[best_ch_idx, "n_clusters"])
        print(f"\n📈 Melhor Calinski-Harabasz: k = {k_ch}")
        print(f"   Score: {metrics_df.loc[best_ch_idx, 'calinski_harabasz']:.2f}")

        # Melhor Davies-Bouldin (menor é melhor)
        best_db_idx = metrics_df["davies_bouldin"].idxmin()
        k_db = int(metrics_df.loc[best_db_idx, "n_clusters"])
        print(f"\n📉 Melhor Davies-Bouldin: k = {k_db}")
        print(f"   Score: {metrics_df.loc[best_db_idx, 'davies_bouldin']:.4f}")

        # Método do cotovelo
        inertias = metrics_df["inertia"].values
        diff2 = np.diff(inertias, n=2)
        k_elbow = int(metrics_df.loc[np.argmax(np.abs(diff2)) + 2, "n_clusters"])
        print(f"\n🔄 Método do Cotovelo: k = {k_elbow}")

        # Visualiza métricas
        self._visualize_metrics(metrics_df)

        return metrics_df

    def _visualize_metrics(self, metrics_df):
        """
        Visualiza as métricas de otimização
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Otimização do Número de Clusters (K) - K-Means\nAnálise de Espessura Epitelial",
            fontsize=16,
            fontweight="bold",
        )

        k_values = metrics_df["n_clusters"]

        # 1. Método do Cotovelo - Inércia
        ax1.plot(k_values, metrics_df["inertia"], "bo-", linewidth=2, markersize=8)
        ax1.set_xlabel("Número de Clusters (k)", fontweight="bold")
        ax1.set_ylabel("Inércia", fontweight="bold")
        ax1.set_title("Método do Cotovelo\nProcure o ponto de inflexão")
        ax1.grid(True, alpha=0.3)

        # 2. Silhouette Score
        ax2.plot(k_values, metrics_df["silhouette"], "go-", linewidth=2, markersize=8)
        ax2.set_xlabel("Número de Clusters (k)", fontweight="bold")
        ax2.set_ylabel("Silhouette Score", fontweight="bold")
        ax2.set_title("Silhouette Score\nMaior é melhor")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(
            y=metrics_df["silhouette"].max(),
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Máximo",
        )
        ax2.legend()

        # 3. Calinski-Harabasz Score
        ax3.plot(
            k_values,
            metrics_df["calinski_harabasz"],
            "ro-",
            linewidth=2,
            markersize=8,
        )
        ax3.set_xlabel("Número de Clusters (k)", fontweight="bold")
        ax3.set_ylabel("Calinski-Harabasz Score", fontweight="bold")
        ax3.set_title("Calinski-Harabasz Score\nMaior é melhor")
        ax3.grid(True, alpha=0.3)

        # 4. Davies-Bouldin Score
        ax4.plot(
            k_values, metrics_df["davies_bouldin"], "mo-", linewidth=2, markersize=8
        )
        ax4.set_xlabel("Número de Clusters (k)", fontweight="bold")
        ax4.set_ylabel("Davies-Bouldin Score", fontweight="bold")
        ax4.set_title("Davies-Bouldin Score\nMenor é melhor")
        ax4.grid(True, alpha=0.3)
        ax4.axhline(
            y=metrics_df["davies_bouldin"].min(),
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Mínimo",
        )
        ax4.legend()

        plt.tight_layout()

        # Salva
        save_path = Path(__file__).parent / "results" / "k_optimization.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n💾 Gráfico de otimização salvo em: {save_path}")

        plt.show()


if __name__ == "__main__":
    print(
        "Este é um módulo de classes. Use o notebook 'analise_clustering.ipynb' para executar a análise."
    )
    print("Ou importe a classe KOptimizer em seus scripts.")
