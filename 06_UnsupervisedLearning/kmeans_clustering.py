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
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class KMeansClusterer:
    def __init__(self, n_clusters=3, random_state=42):
        """
        Inicializa o algoritmo K-Means

        Args:
            n_clusters (int): Número de clusters
            random_state (int): Semente para reprodutibilidade
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.data = None
        self.scaled_data = None
        self.labels = None

    def load_data(self, file_path):
        """
        Carrega os dados do arquivo CSV

        Args:
            file_path (str): Caminho para o arquivo CSV
        """
        self.data = pd.read_csv(file_path)
        print(
            f"Dados carregados: {self.data.shape[0]} amostras, {self.data.shape[1]} características"
        )

    def preprocess_data(self):
        """
        Preprocessa os dados: seleciona apenas as variáveis de interesse e normaliza
        """
        # Seleciona apenas as variáveis de interesse (excluindo ID e Correto)
        features = ["AL", "ACD", "WTW", "K1", "K2"]
        self.features_data = self.data[features].copy()

        # Normaliza os dados
        self.scaled_data = self.scaler.fit_transform(self.features_data)

        print("Dados preprocessados e normalizados")
        print(f"Características utilizadas: {features}")

    def find_optimal_clusters(self, max_clusters=10):
        """
        Encontra o número ótimo de clusters usando método do cotovelo

        Args:
            max_clusters (int): Número máximo de clusters a testar

        Returns:
            dict: Dicionário com as métricas para cada número de clusters
        """
        metrics = {
            "n_clusters": [],
            "inertia": [],
            "silhouette": [],
            "calinski_harabasz": [],
            "davies_bouldin": [],
        }

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)

            metrics["n_clusters"].append(k)
            metrics["inertia"].append(kmeans.inertia_)
            metrics["silhouette"].append(silhouette_score(self.scaled_data, labels))
            metrics["calinski_harabasz"].append(
                calinski_harabasz_score(self.scaled_data, labels)
            )
            metrics["davies_bouldin"].append(
                davies_bouldin_score(self.scaled_data, labels)
            )

        return metrics

    def fit(self):
        """
        Treina o modelo K-Means
        """
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self.labels = self.kmeans.fit_predict(self.scaled_data)

        print(f"K-Means treinado com {self.n_clusters} clusters")

    def evaluate(self):
        """
        Avalia a qualidade do clustering

        Returns:
            dict: Métricas de avaliação
        """
        metrics = {
            "silhouette_score": silhouette_score(self.scaled_data, self.labels),
            "calinski_harabasz_score": calinski_harabasz_score(
                self.scaled_data, self.labels
            ),
            "davies_bouldin_score": davies_bouldin_score(self.scaled_data, self.labels),
            "inertia": self.kmeans.inertia_,
        }

        return metrics

    def get_cluster_profiles(self):
        """
        Retorna o perfil de cada cluster

        Returns:
            pd.DataFrame: DataFrame com estatísticas de cada cluster
        """
        # Adiciona os labels aos dados originais
        data_with_clusters = self.features_data.copy()
        data_with_clusters["Cluster"] = self.labels

        # Calcula estatísticas por cluster
        cluster_stats = (
            data_with_clusters.groupby("Cluster")
            .agg(
                {
                    "AL": ["mean", "std", "min", "max"],
                    "ACD": ["mean", "std", "min", "max"],
                    "WTW": ["mean", "std", "min", "max"],
                    "K1": ["mean", "std", "min", "max"],
                    "K2": ["mean", "std", "min", "max"],
                }
            )
            .round(2)
        )

        # Conta o número de amostras por cluster
        cluster_counts = data_with_clusters["Cluster"].value_counts().sort_index()

        return cluster_stats, cluster_counts

    def plot_clusters_2d(self, feature_x="AL", feature_y="ACD", save_path=None):
        """
        Plota os clusters em 2D usando duas características

        Args:
            feature_x (str): Característica para o eixo X
            feature_y (str): Característica para o eixo Y
            save_path (str): Caminho para salvar o gráfico
        """
        plt.figure(figsize=(10, 8))

        # Plota os pontos coloridos por cluster
        scatter = plt.scatter(
            self.features_data[feature_x],
            self.features_data[feature_y],
            c=self.labels,
            cmap="viridis",
            alpha=0.7,
        )

        # Plota os centroides
        centroids = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        feature_indices = {"AL": 0, "ACD": 1, "WTW": 2, "K1": 3, "K2": 4}

        plt.scatter(
            centroids[:, feature_indices[feature_x]],
            centroids[:, feature_indices[feature_y]],
            c="red",
            marker="x",
            s=200,
            linewidths=3,
            label="Centroides",
        )

        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(f"K-Means Clustering: {feature_x} vs {feature_y}")
        plt.colorbar(scatter, label="Cluster")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def save_results(self, output_path):
        """
        Salva os resultados em um arquivo CSV

        Args:
            output_path (str): Caminho para salvar os resultados
        """
        results = self.data.copy()
        results["KMeans_Cluster"] = self.labels
        results.to_csv(output_path, index=False)
        print(f"Resultados salvos em: {output_path}")


def main():
    # Configurações
    data_path = Path(__file__).parent / "data" / "barrettII_eyes_clustering.csv"
    n_clusters = 3  # Configurável

    # Inicializa o clusterer
    clusterer = KMeansClusterer(n_clusters=n_clusters)

    # Carrega e preprocessa os dados
    clusterer.load_data(data_path)
    clusterer.preprocess_data()

    # Encontra número ótimo de clusters (opcional)
    print("\nEncontrando número ótimo de clusters...")
    metrics = clusterer.find_optimal_clusters()

    # Exibe as métricas
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)

    # Treina o modelo
    clusterer.fit()

    # Avalia o modelo
    evaluation = clusterer.evaluate()
    print(f"\nMétricas de avaliação:")
    for metric, value in evaluation.items():
        print(f"{metric}: {value:.4f}")

    # Obtém perfis dos clusters
    cluster_stats, cluster_counts = clusterer.get_cluster_profiles()
    print(f"\nContagem de amostras por cluster:")
    print(cluster_counts)

    print(f"\nEstatísticas dos clusters:")
    print(cluster_stats)

    # Plota os clusters
    clusterer.plot_clusters_2d("AL", "ACD")
    clusterer.plot_clusters_2d("K1", "K2")

    # Salva os resultados
    output_path = Path(__file__).parent / "results" / "kmeans_results.csv"
    output_path.parent.mkdir(exist_ok=True)
    clusterer.save_results(output_path)


if __name__ == "__main__":
    main()
