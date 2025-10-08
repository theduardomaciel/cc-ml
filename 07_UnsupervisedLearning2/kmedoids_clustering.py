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
import sys

# Configurar encoding para UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")


class KMedoidsEpithelialClusterer:
    def __init__(self, n_clusters=3, random_state=42, method="pam"):
        """
        Inicializa o algoritmo K-Medoids para análise de espessura epitelial

        Args:
            n_clusters (int): Número de clusters
            random_state (int): Semente para reprodutibilidade
            method (str): Método de inicialização ('pam' ou 'alternate')
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
        print(f"Valores ausentes por coluna:\n{self.data.isnull().sum()}")

    def preprocess_data(self):
        """
        Preprocessa os dados: seleciona apenas as variáveis de espessura epitelial e normaliza
        Remove linhas com valores ausentes nas features de interesse
        """
        if self.data is None:
            raise ValueError(
                "Os dados ainda não foram carregados. Use load_data primeiro."
            )

        # Seleciona apenas as variáveis de espessura epitelial
        features = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
        self.features_data = self.data[features].copy()

        # Remove linhas com valores ausentes
        print(f"\nLinhas antes de remover NaN: {len(self.features_data)}")
        self.features_data = self.features_data.dropna()
        print(f"Linhas após remover NaN: {len(self.features_data)}")

        # Normaliza os dados
        self.scaled_data = self.scaler.fit_transform(self.features_data)

        print("\nDados preprocessados e normalizados")
        print(f"Características utilizadas: {features}")
        print(f"Estatísticas descritivas:")
        print(self.features_data.describe())

    def find_optimal_clusters(self, max_clusters=10):
        """
        Encontra o número ótimo de clusters usando diferentes métricas

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

        print(f"\n🔍 Testando de 2 a {max_clusters} clusters...")

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

            print(
                f"  k={k}: Silhouette={metrics['silhouette'][-1]:.3f}, "
                f"Calinski-Harabasz={metrics['calinski_harabasz'][-1]:.2f}, "
                f"Davies-Bouldin={metrics['davies_bouldin'][-1]:.3f}"
            )

        return metrics

    def fit(self):
        """
        Treina o modelo K-Medoids com o número de clusters especificado
        """
        if self.scaled_data is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados. Use preprocess_data primeiro."
            )

        print(
            f"\n🎯 Treinando K-Medoids com {self.n_clusters} clusters (método: {self.method})..."
        )
        self.kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            method=self.method,
        )
        self.labels = self.kmedoids.fit_predict(self.scaled_data)

        # Calcula métricas
        silhouette = silhouette_score(self.scaled_data, self.labels)
        calinski = calinski_harabasz_score(self.scaled_data, self.labels)
        davies = davies_bouldin_score(self.scaled_data, self.labels)

        print(f"\n📊 Métricas do modelo:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski:.2f}")
        print(f"  Davies-Bouldin Score: {davies:.4f}")
        print(f"  Inércia: {self.kmedoids.inertia_:.2f}")

        return self

    def get_cluster_profiles(self):
        """
        Retorna o perfil de cada cluster

        Returns:
            tuple: (DataFrame com estatísticas dos clusters, Series com contagens)
        """
        if self.labels is None:
            raise ValueError("O modelo ainda não foi treinado. Use fit primeiro.")

        # Adiciona labels aos dados
        data_with_clusters = self.features_data.copy()
        data_with_clusters["Cluster"] = self.labels

        # Calcula estatísticas por cluster
        cluster_stats = data_with_clusters.groupby("Cluster").agg(
            ["mean", "std", "min", "max"]
        )

        # Conta número de amostras por cluster
        cluster_counts = data_with_clusters["Cluster"].value_counts().sort_index()

        print(f"\n📈 Distribuição de amostras por cluster:")
        for cluster, count in cluster_counts.items():
            percentage = (count / len(data_with_clusters)) * 100
            print(f"  Cluster {cluster}: {count} amostras ({percentage:.1f}%)")

        return cluster_stats, cluster_counts

    def get_medoids(self):
        """
        Retorna os medoides (pontos representativos) de cada cluster

        Returns:
            DataFrame: Features dos medoides
        """
        if self.kmedoids is None:
            raise ValueError("O modelo ainda não foi treinado. Use fit primeiro.")

        medoid_indices = self.kmedoids.medoid_indices_
        medoids = self.features_data.iloc[medoid_indices]

        print(f"\n🎯 Medoides dos clusters:")
        print(medoids)

        return medoids

    def save_results(self, output_path="results/kmedoids_results.csv"):
        """
        Salva os resultados em um arquivo CSV

        Args:
            output_path (str): Caminho para salvar os resultados
        """
        if self.labels is None:
            raise ValueError("O modelo ainda não foi treinado. Use fit primeiro.")

        # Combina features com labels
        results_df = self.features_data.copy()
        results_df["Cluster"] = self.labels

        # Adiciona as colunas originais de volta
        original_cols = ["Index", "pID", "Age", "Gender", "Eye"]
        for col in original_cols:
            if col in self.data.columns:
                # Filtra apenas os índices que não foram removidos
                results_df[col] = self.data.loc[results_df.index, col].values

        # Salva
        results_dir = Path(output_path).parent
        results_dir.mkdir(exist_ok=True)
        results_df.to_csv(output_path, index=False)

        print(f"\n💾 Resultados salvos em: {output_path}")

        return results_df


if __name__ == "__main__":
    # Exemplo de uso
    clusterer = KMedoidsEpithelialClusterer(n_clusters=3, method="pam")
    clusterer.load_data("data/RTVue_20221110_MLClass.csv")
    clusterer.preprocess_data()

    # Encontra k ótimo
    metrics = clusterer.find_optimal_clusters(max_clusters=10)

    # Treina com k=3
    clusterer.fit()
    cluster_stats, cluster_counts = clusterer.get_cluster_profiles()
    medoids = clusterer.get_medoids()
    clusterer.save_results()
