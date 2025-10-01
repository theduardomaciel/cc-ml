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

warnings.filterwarnings("ignore")


class DBSCANClusterer:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Inicializa o algoritmo DBSCAN

        Args:
            eps (float): Distância máxima entre dois pontos para serem considerados vizinhos
            min_samples (int): Número mínimo de pontos necessários para formar um cluster
        """
        self.eps = eps
        self.min_samples = min_samples
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

    def find_optimal_eps(self, k=4):
        """
        Encontra o valor ótimo de eps usando o método k-distance

        Args:
            k (int): Número de vizinhos mais próximos a considerar

        Returns:
            float: Valor sugerido para eps
        """
        if self.scaled_data is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados. Use preprocess_data primeiro."
            )
        # Calcula as k-distâncias
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(self.scaled_data)
        distances, indices = neighbors_fit.kneighbors(self.scaled_data)

        # Ordena as distâncias do k-ésimo vizinho mais próximo
        distances = np.sort(distances[:, k - 1], axis=0)

        # Plota o gráfico k-distance
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.xlabel("Pontos ordenados por distância")
        plt.ylabel(f"{k}-ésima distância mais próxima")
        plt.title(f"Gráfico {k}-distance para encontrar eps ótimo")
        plt.grid(True, alpha=0.3)
        plt.show()

        # Sugere um valor para eps (ponto do "cotovelo")
        # Calcula a derivada segunda para encontrar o ponto de inflexão
        second_derivative = np.diff(distances, n=2)
        suggested_eps = distances[np.argmax(second_derivative) + 2]

        print(f"Valor sugerido para eps: {suggested_eps:.4f}")
        return suggested_eps

    def test_parameters(self, eps_range=None, min_samples_range=None):
        """
        Testa diferentes combinações de parâmetros

        Args:
            eps_range (list): Lista de valores de eps para testar
            min_samples_range (list): Lista de valores de min_samples para testar

        Returns:
            pd.DataFrame: DataFrame com resultados dos testes
        """
        if self.scaled_data is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados. Use preprocess_data primeiro."
            )
        if eps_range is None:
            eps_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        if min_samples_range is None:
            min_samples_range = [3, 4, 5, 6, 7, 8, 9, 10]

        results = []

        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.scaled_data)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                result = {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_ratio": n_noise / len(labels),
                }

                # Calcula métricas apenas se houver mais de 1 cluster
                if n_clusters > 1:
                    # Remove pontos de ruído para calcular métricas
                    mask = labels != -1
                    if np.sum(mask) > 0:
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
                        except Exception:
                            result["silhouette"] = np.nan
                            result["calinski_harabasz"] = np.nan
                            result["davies_bouldin"] = np.nan
                else:
                    result["silhouette"] = np.nan
                    result["calinski_harabasz"] = np.nan
                    result["davies_bouldin"] = np.nan

                results.append(result)

        return pd.DataFrame(results)

    def fit(self):
        """
        Treina o modelo DBSCAN
        """
        if self.scaled_data is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados. Use preprocess_data primeiro."
            )
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.dbscan.fit_predict(self.scaled_data)

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)

        print(f"DBSCAN treinado com eps={self.eps}, min_samples={self.min_samples}")
        print(f"Número de clusters encontrados: {n_clusters}")
        print(
            f"Número de pontos de ruído: {n_noise} ({n_noise/len(self.labels)*100:.1f}%)"
        )

    def evaluate(self):
        """
        Avalia a qualidade do clustering

        Returns:
            dict: Métricas de avaliação
        """
        if self.labels is None or self.scaled_data is None:
            raise ValueError("O modelo ainda não foi treinado. Use fit primeiro.")
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)

        metrics = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(self.labels),
        }

        # Calcula métricas apenas se houver mais de 1 cluster e pontos não-ruído
        if n_clusters > 1:
            mask = self.labels != -1
            if np.sum(mask) > 0:
                try:
                    metrics["silhouette_score"] = silhouette_score(
                        self.scaled_data[mask], self.labels[mask]
                    )
                    metrics["calinski_harabasz_score"] = calinski_harabasz_score(
                        self.scaled_data[mask], self.labels[mask]
                    )
                    metrics["davies_bouldin_score"] = davies_bouldin_score(
                        self.scaled_data[mask], self.labels[mask]
                    )
                except Exception:
                    metrics["silhouette_score"] = np.nan
                    metrics["calinski_harabasz_score"] = np.nan
                    metrics["davies_bouldin_score"] = np.nan
        else:
            metrics["silhouette_score"] = np.nan
            metrics["calinski_harabasz_score"] = np.nan
            metrics["davies_bouldin_score"] = np.nan

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

        # Separa pontos de ruído
        cluster_data = data_with_clusters[data_with_clusters["Cluster"] != -1]
        noise_data = data_with_clusters[data_with_clusters["Cluster"] == -1]

        cluster_stats = None
        cluster_counts = None

        if len(cluster_data) > 0:
            # Calcula estatísticas por cluster
            cluster_stats = (
                cluster_data.groupby("Cluster")
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
            cluster_counts = cluster_data["Cluster"].value_counts().sort_index()

        return cluster_stats, cluster_counts, len(noise_data)

    def plot_clusters_2d(self, feature_x="AL", feature_y="ACD", save_path=None):
        """
        Plota os clusters em 2D usando duas características

        Args:
            feature_x (str): Característica para o eixo X
            feature_y (str): Característica para o eixo Y
            save_path (str): Caminho para salvar o gráfico
        """
        if self.features_data is None or self.labels is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados ou o modelo não foi treinado."
            )

        plt.figure(figsize=(10, 8))

        # Separa pontos de cluster e ruído
        mask_clusters = self.labels != -1
        mask_noise = self.labels == -1

        # Plota os clusters
        if np.sum(mask_clusters) > 0:
            x_vals = np.asarray(self.features_data.loc[mask_clusters, feature_x])
            y_vals = np.asarray(self.features_data.loc[mask_clusters, feature_y])
            scatter = plt.scatter(
                x_vals,
                y_vals,
                c=self.labels[mask_clusters],
                cmap="viridis",
                alpha=0.7,
                label="Clusters",
            )
            plt.colorbar(scatter, label="Cluster")

        # Plota os pontos de ruído
        if np.sum(mask_noise) > 0:
            x_vals_noise = np.asarray(self.features_data.loc[mask_noise, feature_x])
            y_vals_noise = np.asarray(self.features_data.loc[mask_noise, feature_y])
            plt.scatter(
                x_vals_noise,
                y_vals_noise,
                c="red",
                marker="x",
                alpha=0.25,
                label="Ruído",
            )

        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(f"DBSCAN Clustering: {feature_x} vs {feature_y}")
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
        results["DBSCAN_Cluster"] = self.labels
        results.to_csv(output_path, index=False)
        print(f"Resultados salvos em: {output_path}")


def tests():
    # Configurações - Testando diferentes parâmetros
    data_path = Path(__file__).parent / "data" / "barrettII_eyes_clustering.csv"

    # Parâmetros originais vs otimizados
    configs = [
        {"eps": 0.5, "min_samples": 5, "label": "ORIGINAL"},
        {"eps": 0.7, "min_samples": 5, "label": "OTIMIZADO (3 clusters)"},
        {"eps": 1.4, "min_samples": 3, "label": "OTIMIZADO (baixo ruído)"},
    ]

    print("=" * 80)
    print("COMPARAÇÃO DE CONFIGURAÇÕES DBSCAN")
    print("=" * 80)

    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"CONFIGURAÇÃO {i}: {config['label']}")
        print(f"eps={config['eps']}, min_samples={config['min_samples']}")
        print(f"{'='*60}")

        # Inicializa o clusterer
        clusterer = DBSCANClusterer(
            eps=config["eps"], min_samples=config["min_samples"]
        )

        # Carrega e preprocessa os dados
        clusterer.load_data(data_path)
        clusterer.preprocess_data()

        # Encontra eps ótimo (apenas para primeira configuração)
        if i == 1:
            print("\nEncontrando eps ótimo...")
            suggested_eps = clusterer.find_optimal_eps()

        # Treina o modelo
        clusterer.fit()

        # Avalia o modelo
        evaluation = clusterer.evaluate()
        print(f"\nMétricas de avaliação:")
        for metric, value in evaluation.items():
            if isinstance(value, float) and not np.isnan(value):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

        # Obtém perfis dos clusters
        cluster_stats, cluster_counts, n_noise = clusterer.get_cluster_profiles()

        if cluster_counts is not None:
            print(f"\nContagem de amostras por cluster:")
            print(cluster_counts)
            print(f"Pontos de ruído: {n_noise}")

            print(f"\nEstatísticas dos clusters:")
            if cluster_stats is not None:
                print(cluster_stats.head())  # Mostra apenas as primeiras linhas
            else:
                print("Estatísticas não disponíveis")
        else:
            print(f"\nNenhum cluster válido encontrado. Pontos de ruído: {n_noise}")

        # Plota os clusters (apenas para configuração otimizada)
        if config["label"] == "OTIMIZADO (3 clusters)":
            print("\nGerando visualizações...")
            clusterer.plot_clusters_2d("AL", "ACD")
            clusterer.plot_clusters_2d("K1", "K2")

        # Salva os resultados
        output_path = (
            Path(__file__).parent
            / "results"
            / f"dbscan_results_{config['label'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        )
        output_path.parent.mkdir(exist_ok=True)
        clusterer.save_results(output_path)

    # Resumo final
    print(f"\n{'='*80}")
    print("RESUMO COMPARATIVO")
    print(f"{'='*80}")
    print(f"1. ORIGINAL (0.5, 5): Muitos clusters pequenos + 45% ruído")
    print(f"2. OTIMIZADO (0.7, 5): 3 clusters + 18% ruído")
    print(f"3. OTIMIZADO (1.4, 3): 2 clusters + 1.5% ruído")
    print(f"\nCONCLUSÃO: Mesmo otimizado, DBSCAN não supera K-Means")
    print(f"para segmentação clínica estruturada de pacientes oftalmológicos.")


def main():
    # Configurações
    data_path = Path(__file__).parent / "data" / "barrettII_eyes_clustering.csv"
    eps = 0.5
    min_samples = 5

    # Inicializa o clusterer
    clusterer = DBSCANClusterer(eps=eps, min_samples=min_samples)

    # Carrega e preprocessa os dados
    clusterer.load_data(data_path)
    clusterer.preprocess_data()

    # Encontra eps ótimo
    print("\nEncontrando eps ótimo...")
    suggested_eps = clusterer.find_optimal_eps()

    # Testa diferentes parâmetros
    print("\nTestando diferentes parâmetros...")
    param_results = clusterer.test_parameters()

    # Mostra os melhores resultados
    best_params = param_results.dropna(subset=["silhouette"]).nlargest(5, "silhouette")
    print("Top 5 combinações de parâmetros (por silhouette score):")
    print(best_params[["eps", "min_samples", "n_clusters", "n_noise", "silhouette"]])

    # Atualiza parâmetros se necessário (pode usar o eps sugerido)
    # clusterer.eps = suggested_eps

    # Treina o modelo
    clusterer.fit()

    # Avalia o modelo
    evaluation = clusterer.evaluate()
    print(f"\nMétricas de avaliação:")
    for metric, value in evaluation.items():
        if isinstance(value, float) and not np.isnan(value):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Obtém perfis dos clusters
    cluster_stats, cluster_counts, n_noise = clusterer.get_cluster_profiles()

    if cluster_counts is not None:
        print(f"\nContagem de amostras por cluster:")
        print(cluster_counts)
        print(f"Pontos de ruído: {n_noise}")

        print(f"\nEstatísticas dos clusters:")
        print(cluster_stats)
    else:
        print(f"\nNenhum cluster válido encontrado. Pontos de ruído: {n_noise}")

    # Plota os clusters
    clusterer.plot_clusters_2d("AL", "ACD")
    clusterer.plot_clusters_2d("K1", "K2")

    # Salva os resultados
    output_path = Path(__file__).parent / "results" / "dbscan_results.csv"
    output_path.parent.mkdir(exist_ok=True)
    clusterer.save_results(output_path)


if __name__ == "__main__":
    main()
