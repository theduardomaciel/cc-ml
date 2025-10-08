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


class KMeansEpithelialClusterer:
    def __init__(self, n_clusters=3, random_state=42):
        """
        Inicializa o algoritmo K-Means para análise de espessura epitelial

        Args:
            n_clusters (int): Número de clusters
            random_state (int): Semente para reprodutibilidade
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
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
        # C=central, S=superior, ST=superior temporal, T=temporal,
        # IT=inferior temporal, I=inferior, IN=inferior nasal, N=nasal, SN=superior nasal
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

            print(
                f"  k={k}: Silhouette={metrics['silhouette'][-1]:.3f}, "
                f"Calinski-Harabasz={metrics['calinski_harabasz'][-1]:.2f}, "
                f"Davies-Bouldin={metrics['davies_bouldin'][-1]:.3f}"
            )

        return metrics

    def fit(self):
        """
        Treina o modelo K-Means com o número de clusters especificado
        """
        if self.scaled_data is None:
            raise ValueError(
                "Os dados ainda não foram preprocessados. Use preprocess_data primeiro."
            )

        print(f"\n🎯 Treinando K-Means com {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self.labels = self.kmeans.fit_predict(self.scaled_data)

        # Calcula métricas
        silhouette = silhouette_score(self.scaled_data, self.labels)
        calinski = calinski_harabasz_score(self.scaled_data, self.labels)
        davies = davies_bouldin_score(self.scaled_data, self.labels)

        print(f"\n📊 Métricas do modelo:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski:.2f}")
        print(f"  Davies-Bouldin Score: {davies:.4f}")
        print(f"  Inércia: {self.kmeans.inertia_:.2f}")

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

    def visualize_clusters(self, save_path=None):
        """
        Visualiza os clusters encontrados

        Args:
            save_path (str, optional): Caminho para salvar a figura
        """
        if self.labels is None:
            raise ValueError("O modelo ainda não foi treinado. Use fit primeiro.")

        # Adiciona labels aos dados
        data_with_clusters = self.features_data.copy()
        data_with_clusters["Cluster"] = self.labels

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(
            "Distribuição de Espessura Epitelial por Cluster\nAnálise de Perfis de Olhos",
            fontsize=16,
            fontweight="bold",
        )

        features = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
        colors = sns.color_palette("husl", self.n_clusters)

        for idx, feature in enumerate(features):
            ax = axes[idx // 3, idx % 3]

            for cluster in range(self.n_clusters):
                cluster_data = data_with_clusters[
                    data_with_clusters["Cluster"] == cluster
                ][feature]
                ax.hist(
                    cluster_data,
                    alpha=0.6,
                    label=f"Cluster {cluster}",
                    bins=20,
                    color=colors[cluster],
                )

            ax.set_xlabel("Espessura (μm)")
            ax.set_ylabel("Frequência")
            ax.set_title(f"Região: {feature}", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\n💾 Figura salva em: {save_path}")

        plt.show()

    def visualize_cluster_means(self, save_path=None):
        """
        Visualiza as médias de espessura por região para cada cluster

        Args:
            save_path (str, optional): Caminho para salvar a figura
        """
        if self.labels is None:
            raise ValueError("O modelo ainda não foi treinado. Use fit primeiro.")

        cluster_stats, _ = self.get_cluster_profiles()

        # Extrai médias
        means_data = []
        for cluster in range(self.n_clusters):
            cluster_means = [
                cluster_stats.loc[cluster, (feature, "mean")]
                for feature in ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
            ]
            means_data.append(cluster_means)

        # Cria visualização
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            "Perfil Médio de Espessura Epitelial por Cluster",
            fontsize=16,
            fontweight="bold",
        )

        features = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
        x = np.arange(len(features))
        width = 0.25
        colors = sns.color_palette("husl", self.n_clusters)

        # Gráfico de barras
        for i, (cluster_means, color) in enumerate(zip(means_data, colors)):
            offset = width * (i - 1)
            ax1.bar(
                x + offset,
                cluster_means,
                width,
                label=f"Cluster {i}",
                alpha=0.8,
                color=color,
            )

        ax1.set_xlabel("Região do Olho", fontweight="bold")
        ax1.set_ylabel("Espessura Média (μm)", fontweight="bold")
        ax1.set_title("Comparação de Espessuras por Região")
        ax1.set_xticks(x)
        ax1.set_xticklabels(features)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]

        ax2 = plt.subplot(1, 2, 2, projection="polar")

        for i, (cluster_means, color) in enumerate(zip(means_data, colors)):
            values = cluster_means + [cluster_means[0]]
            ax2.plot(
                angles, values, "o-", linewidth=2, label=f"Cluster {i}", color=color
            )
            ax2.fill(angles, values, alpha=0.25, color=color)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(features)
        ax2.set_title("Radar Plot - Perfil Espacial", fontweight="bold", pad=20)
        ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\n💾 Figura salva em: {save_path}")

        plt.show()

    def save_results(self, output_path):
        """
        Salva os resultados da clusterização em CSV

        Args:
            output_path (str): Caminho para salvar o arquivo CSV
        """
        if self.labels is None:
            raise ValueError("O modelo ainda não foi treinado. Use fit primeiro.")

        results = self.features_data.copy()
        results["Cluster"] = self.labels

        # Adiciona informações originais (se disponíveis)
        if self.data is not None:
            original_cols = ["Index", "pID", "Age", "Gender", "Eye"]
            available_cols = [col for col in original_cols if col in self.data.columns]
            for col in available_cols:
                # Garante que os índices correspondem
                results[col] = self.data.loc[results.index, col].values

        results.to_csv(output_path, index=False)
        print(f"\n💾 Resultados salvos em: {output_path}")


if __name__ == "__main__":
    print(
        "Este é um módulo de classes. Use o notebook 'analise_clustering.ipynb' para executar a análise."
    )
    print("Ou importe a classe KMeansEpithelialClusterer em seus scripts.")
