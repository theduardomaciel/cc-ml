import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
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


class KMedoidsClusterer:
    def __init__(self, n_clusters=3, random_state=42, method="pam"):
        """
        Inicializa o algoritmo K-Medoids

        Args:
            n_clusters (int): Número de clusters
            random_state (int): Semente para reprodutibilidade
            method (str): Método para K-Medoids ('pam' ou 'alternate')
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.method = method
        self.kmedoids = None
        self.scaler = StandardScaler()
        self.data = None
        self.features_data = None
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
        if self.data is None:
            raise ValueError(
                "Os dados ainda não foram carregados. Use load_data primeiro."
            )
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
        if self.scaled_data is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados. Use preprocess_data primeiro."
            )
        metrics = {
            "n_clusters": [],
            "inertia": [],
            "silhouette": [],
            "calinski_harabasz": [],
            "davies_bouldin": [],
        }

        for k in range(2, max_clusters + 1):
            kmedoids = KMedoids(
                n_clusters=k, random_state=self.random_state, method=self.method
            )
            labels = kmedoids.fit_predict(self.scaled_data)

            metrics["n_clusters"].append(k)
            metrics["inertia"].append(kmedoids.inertia_)
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
        Treina o modelo K-Medoids
        """
        if self.scaled_data is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados. Use preprocess_data primeiro."
            )
        self.kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            method=self.method,
        )
        self.labels = self.kmedoids.fit_predict(self.scaled_data)

        print(
            f"K-Medoids treinado com {self.n_clusters} clusters usando método '{self.method}'"
        )

    def evaluate(self):
        """
        Avalia a qualidade do clustering

        Returns:
            dict: Métricas de avaliação
        """
        if self.scaled_data is None or self.labels is None or self.kmedoids is None:
            raise ValueError("O modelo ainda não foi treinado. Use fit primeiro.")
        metrics = {
            "silhouette_score": silhouette_score(self.scaled_data, self.labels),
            "calinski_harabasz_score": calinski_harabasz_score(
                self.scaled_data, self.labels
            ),
            "davies_bouldin_score": davies_bouldin_score(self.scaled_data, self.labels),
            "inertia": self.kmedoids.inertia_,
        }

        return metrics

    def get_cluster_profiles(self):
        """
        Retorna o perfil de cada cluster

        Returns:
            pd.DataFrame: DataFrame com estatísticas de cada cluster
        """
        if self.features_data is None or self.labels is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados ou o modelo não foi treinado."
            )
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

    def get_medoids_info(self):
        """
        Retorna informações sobre os medoides

        Returns:
            pd.DataFrame: DataFrame com informações dos medoides
        """
        if self.kmedoids is None or self.features_data is None:
            raise ValueError(
                "O modelo ainda não foi treinado ou os dados não foram preprocessados."
            )
        medoid_indices = self.kmedoids.medoid_indices_
        medoids_data = self.features_data.iloc[medoid_indices].copy()
        medoids_data["Cluster"] = range(self.n_clusters)
        medoids_data["Medoid_Index"] = medoid_indices

        return medoids_data

    def plot_clusters_2d(self, feature_x="AL", feature_y="ACD", save_path=None):
        """
        Plota os clusters em 2D usando duas características

        Args:
            feature_x (str): Característica para o eixo X
            feature_y (str): Característica para o eixo Y
            save_path (str): Caminho para salvar o gráfico
        """
        if self.features_data is None or self.labels is None or self.kmedoids is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados ou o modelo não foi treinado."
            )
        plt.figure(figsize=(10, 8))

        # Plota os pontos coloridos por cluster
        scatter = plt.scatter(
            self.features_data[feature_x],
            self.features_data[feature_y],
            c=self.labels,
            cmap="viridis",
            alpha=0.7,
        )

        # Plota os medoides
        medoids_info = self.get_medoids_info()
        plt.scatter(
            medoids_info[feature_x],
            medoids_info[feature_y],
            c="red",
            marker="x",
            s=200,
            linewidths=3,
            label="Medoides",
        )

        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(f"K-Medoids Clustering: {feature_x} vs {feature_y}")
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
        if self.data is None or self.labels is None:
            raise ValueError(
                "Os dados ainda não foram carregados ou o modelo não foi treinado."
            )
        results = self.data.copy()
        results["KMedoids_Cluster"] = self.labels
        results.to_csv(output_path, index=False)
        print(f"Resultados salvos em: {output_path}")


def main():
    # Configurações
    data_path = Path(__file__).parent / "data" / "barrettII_eyes_clustering.csv"
    n_clusters = 3  # Configurável
    method = "pam"  # Configurável: 'pam' ou 'alternate'

    # Inicializa o clusterer
    clusterer = KMedoidsClusterer(n_clusters=n_clusters, method=method)

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

    # Obtém informações dos medoides
    medoids_info = clusterer.get_medoids_info()
    print(f"\nInformações dos medoides:")
    print(medoids_info)

    # Plota os clusters
    clusterer.plot_clusters_2d("AL", "ACD")
    clusterer.plot_clusters_2d("K1", "K2")

    # Salva os resultados
    output_path = Path(__file__).parent / "results" / "kmedoids_results.csv"
    output_path.parent.mkdir(exist_ok=True)
    clusterer.save_results(output_path)


if __name__ == "__main__":
    main()
