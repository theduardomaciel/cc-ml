import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys

# Configurar encoding para UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")


class DBSCANEpithelialClusterer:
    def __init__(self, eps=0.5, min_samples=5, random_state=42):
        """
        Inicializa o algoritmo DBSCAN para análise de espessura epitelial

        Args:
            eps (float): Distância máxima entre dois pontos para serem considerados vizinhos
            min_samples (int): Número mínimo de amostras em uma vizinhança para um ponto ser considerado core
            random_state (int): Semente para reprodutibilidade
        """
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.dbscan = None
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

    def find_optimal_eps(self, k=5):
        """
        Encontra o valor ótimo de eps usando o método do cotovelo com k-distance

        Args:
            k (int): Número de vizinhos para calcular a distância

        Returns:
            float: Valor sugerido de eps
        """
        if self.scaled_data is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados. Use preprocess_data primeiro."
            )

        print(f"\n🔍 Calculando k-distance para k={k}...")

        # Calcula as k distâncias mais próximas
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(self.scaled_data)
        distances, indices = neighbors_fit.kneighbors(self.scaled_data)

        # Ordena as distâncias
        distances = np.sort(distances[:, k - 1], axis=0)

        # Visualiza o gráfico de k-distance
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.xlabel("Pontos ordenados")
        plt.ylabel(f"{k}-distance")
        plt.title(f"K-distance Graph (k={k})\nProcure pelo 'cotovelo' para determinar eps")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Salva a figura
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / "dbscan_eps_optimization.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Sugestão de eps (90º percentil como heurística)
        suggested_eps = np.percentile(distances, 90)
        print(f"📊 Valor sugerido de eps (90º percentil): {suggested_eps:.4f}")

        return suggested_eps

    def optimize_parameters(self, eps_range=None, min_samples_range=None):
        """
        Testa diferentes combinações de eps e min_samples

        Args:
            eps_range (list): Lista de valores de eps para testar
            min_samples_range (list): Lista de valores de min_samples para testar

        Returns:
            DataFrame: Resultados da otimização
        """
        if self.scaled_data is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados. Use preprocess_data primeiro."
            )

        if eps_range is None:
            eps_range = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        if min_samples_range is None:
            min_samples_range = [3, 5, 10, 15, 20]

        results = []

        print(f"\n🔍 Testando combinações de parâmetros...")

        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.scaled_data)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_pct = (n_noise / len(labels)) * 100

                result = {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_pct": noise_pct,
                }

                # Calcula métricas apenas se houver pelo menos 2 clusters
                if n_clusters >= 2 and n_noise < len(labels):
                    # Remove ruído para cálculo de métricas
                    mask = labels != -1
                    if sum(mask) > 0:
                        try:
                            result["silhouette"] = silhouette_score(
                                self.scaled_data[mask], labels[mask]
                            )
                            result["calinski_harabasz"] = calinski_harabasz_score(
                                self.scaled_data[mask], labels[mask]
                            )
                            result["davies_bouldin"] = davies_bouldin_score(
                                self.scaled_data[mask], labels[mask]
                            )
                        except:
                            result["silhouette"] = np.nan
                            result["calinski_harabasz"] = np.nan
                            result["davies_bouldin"] = np.nan
                else:
                    result["silhouette"] = np.nan
                    result["calinski_harabasz"] = np.nan
                    result["davies_bouldin"] = np.nan

                results.append(result)

                silhouette_str = f"{result['silhouette']:.3f}" if not np.isnan(result['silhouette']) else 'N/A'
                print(
                    f"  eps={eps:.2f}, min_samples={min_samples}: "
                    f"{n_clusters} clusters, {n_noise} ruídos ({noise_pct:.1f}%), "
                    f"Silhouette={silhouette_str}"
                )

        return pd.DataFrame(results)

    def fit(self):
        """
        Treina o modelo DBSCAN com os parâmetros especificados
        """
        if self.scaled_data is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados. Use preprocess_data primeiro."
            )

        print(f"\n🎯 Treinando DBSCAN com eps={self.eps}, min_samples={self.min_samples}...")
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.dbscan.fit_predict(self.scaled_data)

        # Calcula estatísticas
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        noise_pct = (n_noise / len(self.labels)) * 100

        print(f"\n📊 Resultados do DBSCAN:")
        print(f"  Número de clusters: {n_clusters}")
        print(f"  Número de ruídos: {n_noise} ({noise_pct:.2f}%)")

        # Calcula métricas apenas se houver pelo menos 2 clusters e pontos não-ruído
        if n_clusters >= 2 and n_noise < len(self.labels):
            mask = self.labels != -1
            if sum(mask) > 0:
                try:
                    silhouette = silhouette_score(self.scaled_data[mask], self.labels[mask])
                    calinski = calinski_harabasz_score(self.scaled_data[mask], self.labels[mask])
                    davies = davies_bouldin_score(self.scaled_data[mask], self.labels[mask])

                    print(f"\n📊 Métricas do modelo (sem ruído):")
                    print(f"  Silhouette Score: {silhouette:.4f}")
                    print(f"  Calinski-Harabasz Score: {calinski:.2f}")
                    print(f"  Davies-Bouldin Score: {davies:.4f}")
                except Exception as e:
                    print(f"⚠️ Erro ao calcular métricas: {e}")

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

        # Separa ruído dos clusters
        cluster_counts = data_with_clusters["Cluster"].value_counts().sort_index()

        print(f"\n📈 Distribuição de amostras por cluster:")
        for cluster, count in cluster_counts.items():
            percentage = (count / len(data_with_clusters)) * 100
            label = "Ruído" if cluster == -1 else f"Cluster {cluster}"
            print(f"  {label}: {count} amostras ({percentage:.1f}%)")

        # Calcula estatísticas por cluster (excluindo ruído)
        data_no_noise = data_with_clusters[data_with_clusters["Cluster"] != -1]
        if len(data_no_noise) > 0:
            cluster_stats = data_no_noise.groupby("Cluster").agg(["mean", "std", "min", "max"])
        else:
            cluster_stats = None

        return cluster_stats, cluster_counts

    def save_results(self, output_path="results/dbscan_results.csv"):
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
    clusterer = DBSCANEpithelialClusterer()
    clusterer.load_data("data/RTVue_20221110_MLClass.csv")
    clusterer.preprocess_data()

    # Encontra eps ótimo
    suggested_eps = clusterer.find_optimal_eps(k=5)

    # Otimiza parâmetros
    optimization_results = clusterer.optimize_parameters()
    print("\n📊 Top 10 melhores configurações:")
    print(
        optimization_results.dropna(subset=["silhouette"])
        .nlargest(10, "silhouette")[
            ["eps", "min_samples", "n_clusters", "noise_pct", "silhouette"]
        ]
    )

    # Treina com os melhores parâmetros
    best_config = optimization_results.dropna(subset=["silhouette"]).nlargest(1, "silhouette").iloc[0]
    clusterer.eps = best_config["eps"]
    clusterer.min_samples = int(best_config["min_samples"])

    clusterer.fit()
    cluster_stats, cluster_counts = clusterer.get_cluster_profiles()
    clusterer.save_results()
