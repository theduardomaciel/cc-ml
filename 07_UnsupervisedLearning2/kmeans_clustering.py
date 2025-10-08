"""
Módulo de Clusterização K-Means para Perfis de Espessura Epitelial
Seguindo a metodologia KDD (Knowledge Discovery in Databases)
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


class KMeansEpithelialClusterer:
    """
    Classe para clusterização de padrões de espessura epitelial usando K-Means.
    Implementa as etapas do KDD: Seleção, Pré-processamento, Transformação,
    Mineração e Interpretação.
    """

    def __init__(self, n_clusters=3, random_state=42):
        """
        Inicializa o clusterizador.

        Args:
            n_clusters (int): Número de clusters desejado
            random_state (int): Semente para reprodutibilidade
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.data = None
        self.data_normalized = None
        self.features = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
        self.feature_names = {
            "C": "Central",
            "S": "Superior",
            "ST": "Superior Temporal",
            "T": "Temporal",
            "IT": "Inferior Temporal",
            "I": "Inferior",
            "IN": "Inferior Nasal",
            "N": "Nasal",
            "SN": "Superior Nasal",
        }

    def load_and_select_data(self, filepath):
        """
        KDD Etapa 1: Seleção de Dados
        Carrega e seleciona as variáveis relevantes.

        Args:
            filepath (str): Caminho para o arquivo CSV

        Returns:
            pd.DataFrame: Dados carregados
        """
        print("=" * 60)
        print("KDD ETAPA 1: SELEÇÃO DE DADOS")
        print("=" * 60)

        df = pd.read_csv(filepath)
        print(f"✓ Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas")
        print(f"✓ Variáveis selecionadas para análise: {', '.join(self.features)}")

        return df

    def preprocess_data(self, df):
        """
        KDD Etapa 2: Pré-processamento
        Remove valores ausentes e dados inconsistentes.

        Args:
            df (pd.DataFrame): Dataset original

        Returns:
            pd.DataFrame: Dados pré-processados
        """
        print("\n" + "=" * 60)
        print("KDD ETAPA 2: PRÉ-PROCESSAMENTO")
        print("=" * 60)

        initial_count = len(df)

        # Remover valores ausentes
        df_clean = df[self.features].dropna()

        removed_count = initial_count - len(df_clean)
        print(f"✓ Registros removidos (valores ausentes): {removed_count}")
        print(f"✓ Registros restantes: {len(df_clean)}")

        # Estatísticas descritivas
        print("\nEstatísticas Descritivas:")
        print(df_clean.describe().round(2))

        self.data = df_clean
        return df_clean

    def transform_data(self):
        """
        KDD Etapa 3: Transformação
        Normaliza os dados usando StandardScaler.

        Returns:
            np.ndarray: Dados normalizados
        """
        print("\n" + "=" * 60)
        print("KDD ETAPA 3: TRANSFORMAÇÃO")
        print("=" * 60)

        self.data_normalized = self.scaler.fit_transform(self.data)

        print("✓ Normalização aplicada (StandardScaler)")
        print(f"✓ Forma dos dados normalizados: {self.data_normalized.shape}")
        print(f"✓ Média após normalização: {self.data_normalized.mean():.6f}")
        print(f"✓ Desvio padrão após normalização: {self.data_normalized.std():.6f}")

        return self.data_normalized

    def mine_data(self):
        """
        KDD Etapa 4: Mineração de Dados
        Aplica o algoritmo K-Means.

        Returns:
            np.ndarray: Labels dos clusters
        """
        print("\n" + "=" * 60)
        print("KDD ETAPA 4: MINERAÇÃO DE DADOS (K-MEANS)")
        print("=" * 60)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        )

        labels = self.kmeans.fit_predict(self.data_normalized)

        print(f"✓ Algoritmo: K-Means")
        print(f"✓ Número de clusters: {self.n_clusters}")
        print(f"✓ Número de inicializações: 10")
        print(f"✓ Iterações até convergência: {self.kmeans.n_iter_}")
        print(f"✓ Inércia: {self.kmeans.inertia_:.2f}")

        return labels

    def evaluate_clustering(self, labels):
        """
        Avalia a qualidade da clusterização usando múltiplas métricas.

        Args:
            labels (np.ndarray): Labels dos clusters

        Returns:
            dict: Dicionário com as métricas
        """
        print("\n" + "=" * 60)
        print("AVALIAÇÃO DA CLUSTERIZAÇÃO")
        print("=" * 60)

        metrics = {
            "silhouette": silhouette_score(self.data_normalized, labels),
            "calinski_harabasz": calinski_harabasz_score(self.data_normalized, labels),
            "davies_bouldin": davies_bouldin_score(self.data_normalized, labels),
            "inertia": self.kmeans.inertia_,
        }

        print(f"✓ Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"  (Varia de -1 a 1, quanto maior melhor)")
        print(f"\n✓ Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}")
        print(f"  (Quanto maior melhor)")
        print(f"\n✓ Davies-Bouldin Score: {metrics['davies_bouldin']:.4f}")
        print(f"  (Quanto menor melhor)")
        print(f"\n✓ Inércia: {metrics['inertia']:.2f}")
        print(f"  (Soma das distâncias quadradas aos centróides)")

        return metrics

    def interpret_results(self, labels):
        """
        KDD Etapa 5: Interpretação/Avaliação
        Analisa e interpreta os clusters formados.

        Args:
            labels (np.ndarray): Labels dos clusters

        Returns:
            pd.DataFrame: DataFrame com informações dos clusters
        """
        print("\n" + "=" * 60)
        print("KDD ETAPA 5: INTERPRETAÇÃO/AVALIAÇÃO")
        print("=" * 60)

        # Adicionar labels aos dados originais
        df_result = self.data.copy()
        df_result["Cluster"] = labels

        # Análise por cluster
        cluster_info = []

        for cluster_id in range(self.n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id]

            info = {
                "Cluster": f"Cluster {cluster_id}",
                "Tamanho": len(cluster_data),
                "Percentual": f"{len(cluster_data) / len(df_result) * 100:.1f}%",
            }

            # Médias de cada região
            for feature in self.features:
                info[f"{feature}_média"] = cluster_data[feature].mean()
                info[f"{feature}_std"] = cluster_data[feature].std()

            cluster_info.append(info)

        df_cluster_info = pd.DataFrame(cluster_info)

        # Exibir informações
        print("\nDistribuição dos Clusters:")
        for _, row in df_cluster_info.iterrows():
            print(f"  {row['Cluster']}: {row['Tamanho']} olhos ({row['Percentual']})")

        print("\nPerfis Médios de Espessura Epitelial (μm):")
        print("-" * 60)

        for cluster_id in range(self.n_clusters):
            print(f"\n{df_cluster_info.iloc[cluster_id]['Cluster']}:")
            for feature in self.features:
                mean_val = df_cluster_info.iloc[cluster_id][f"{feature}_média"]
                std_val = df_cluster_info.iloc[cluster_id][f"{feature}_std"]
                print(
                    f"  {self.feature_names[feature]:20s}: {mean_val:5.1f} ± {std_val:4.1f} μm"
                )

        return df_result, df_cluster_info

    def save_results(self, df_result, output_path="results/kmeans_results.csv"):
        """
        Salva os resultados da clusterização.

        Args:
            df_result (pd.DataFrame): DataFrame com os clusters
            output_path (str): Caminho para salvar o arquivo
        """
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_result.to_csv(output_path, index=False)
        print(f"\n✓ Resultados salvos em: {output_path}")

    def fit(self, filepath):
        """
        Executa todo o pipeline KDD.

        Args:
            filepath (str): Caminho para o arquivo CSV

        Returns:
            tuple: (df_result, df_cluster_info, metrics)
        """
        # Etapa 1: Seleção
        df = self.load_and_select_data(filepath)

        # Etapa 2: Pré-processamento
        df_clean = self.preprocess_data(df)

        # Etapa 3: Transformação
        self.transform_data()

        # Etapa 4: Mineração
        labels = self.mine_data()

        # Avaliação
        metrics = self.evaluate_clustering(labels)

        # Etapa 5: Interpretação
        df_result, df_cluster_info = self.interpret_results(labels)

        return df_result, df_cluster_info, metrics


if __name__ == "__main__":
    # Exemplo de uso
    clusterer = KMeansEpithelialClusterer(n_clusters=3)
    df_result, df_cluster_info, metrics = clusterer.fit(
        "data/RTVue_20221110_MLClass.csv"
    )
    clusterer.save_results(df_result)
