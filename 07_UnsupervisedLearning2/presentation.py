import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from kmeans_clustering import KMeansEpithelialClusterer
import warnings

warnings.filterwarnings("ignore")

# Configurar estilo profissional
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ClientPresentation:
    """
    Classe para gerar apresentação profissional dos resultados de clusterização
    para o cliente
    """

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.clusterer = None
        self.cluster_stats = None
        self.cluster_counts = None
        self.data_with_clusters = None

    def setup_and_train(self, data_path):
        """
        Configura, treina o modelo e prepara os dados
        """
        print("=" * 70)
        print("PREPARANDO APRESENTAÇÃO PARA O CLIENTE")
        print("=" * 70)

        # Inicializa e treina
        self.clusterer = KMeansEpithelialClusterer(
            n_clusters=self.n_clusters, random_state=42
        )
        self.clusterer.load_data(data_path)
        self.clusterer.preprocess_data()
        self.clusterer.fit()

        # Obtém perfis
        self.cluster_stats, self.cluster_counts = self.clusterer.get_cluster_profiles()

        # Prepara dados com clusters
        self.data_with_clusters = self.clusterer.features_data.copy()
        self.data_with_clusters["Cluster"] = self.clusterer.labels

        print("\n✅ Modelo treinado e dados preparados!")

    def create_executive_summary(self):
        """
        Cria resumo executivo com visão geral dos perfis encontrados
        """
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        fig.suptitle(
            "PERFIS DE ESPESSURA EPITELIAL - RESUMO EXECUTIVO\nAnálise de Padrões em Mapas Epiteliais Oculares",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # 1. Distribuição de Pacientes por Perfil
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
        wedges, texts, autotexts = ax1.pie(
            self.cluster_counts.values,
            labels=[f"Perfil {i+1}" for i in range(self.n_clusters)],
            autopct="%1.1f%%",
            colors=colors[: self.n_clusters],
            startangle=90,
            explode=[0.05] * self.n_clusters,
        )

        ax1.set_title(
            "Distribuição de Olhos por Perfil",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        for i, (count, autotext) in enumerate(
            zip(self.cluster_counts.values, autotexts)
        ):
            autotext.set_text(f"{autotext.get_text()}\n({count} olhos)")
            autotext.set_fontsize(9)
            autotext.set_fontweight("bold")
            autotext.set_color("white")

        # 2. Radar Chart - Perfil Comparativo
        ax2 = fig.add_subplot(gs[0, 1:], projection="polar")

        categories = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        for cluster in range(self.n_clusters):
            values = [
                self.cluster_stats.loc[cluster, (cat, "mean")] for cat in categories
            ]
            values += values[:1]
            ax2.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=f"Perfil {cluster+1}",
                color=colors[cluster],
            )
            ax2.fill(angles, values, alpha=0.15, color=colors[cluster])

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, size=10)
        ax2.set_title(
            "Perfil Espacial de Espessura por Região\n(valores em μm)",
            fontsize=13,
            fontweight="bold",
            pad=20,
        )
        ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax2.grid(True)

        # 3. Comparação de Médias por Região
        ax3 = fig.add_subplot(gs[1, :])

        regions = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
        x = np.arange(len(regions))
        width = 0.25

        for i in range(self.n_clusters):
            means = [self.cluster_stats.loc[i, (region, "mean")] for region in regions]
            offset = width * (i - (self.n_clusters - 1) / 2)
            ax3.bar(
                x + offset,
                means,
                width,
                label=f"Perfil {i+1}",
                alpha=0.8,
                color=colors[i],
            )

        ax3.set_xlabel("Região do Olho", fontweight="bold", fontsize=12)
        ax3.set_ylabel("Espessura Média (μm)", fontweight="bold", fontsize=12)
        ax3.set_title(
            "Comparação de Espessura Média por Região",
            fontweight="bold",
            fontsize=13,
            pad=15,
        )
        ax3.set_xticks(x)
        ax3.set_xticklabels(regions)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        # Adiciona linha de referência
        overall_mean = np.mean(
            [self.cluster_stats.loc[:, (region, "mean")].mean() for region in regions]
        )
        ax3.axhline(
            y=overall_mean,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Média Geral ({overall_mean:.1f} μm)",
        )

        plt.tight_layout()

        # Salva
        save_path = Path(__file__).parent / "results" / "01_executive_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n💾 Resumo executivo salvo em: {save_path}")

        plt.show()

    def create_detailed_profiles(self):
        """
        Cria visualização detalhada de cada perfil encontrado
        """
        fig, axes = plt.subplots(self.n_clusters, 2, figsize=(16, 5 * self.n_clusters))
        if self.n_clusters == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            "ANÁLISE DETALHADA DOS PERFIS DE ESPESSURA EPITELIAL",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]

        for cluster in range(self.n_clusters):
            # Perfil espacial individual
            ax1 = axes[cluster, 0]

            categories = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
            means = [
                self.cluster_stats.loc[cluster, (cat, "mean")] for cat in categories
            ]
            stds = [self.cluster_stats.loc[cluster, (cat, "std")] for cat in categories]

            ax1.bar(categories, means, alpha=0.7, color=colors[cluster], label="Média")
            ax1.errorbar(
                categories,
                means,
                yerr=stds,
                fmt="none",
                ecolor="black",
                capsize=5,
                label="Desvio Padrão",
            )

            ax1.set_xlabel("Região do Olho", fontweight="bold")
            ax1.set_ylabel("Espessura (μm)", fontweight="bold")
            ax1.set_title(
                f"Perfil {cluster+1}: Espessura por Região ({self.cluster_counts[cluster]} olhos)",
                fontweight="bold",
            )
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis="y")

            # Boxplot das distribuições
            ax2 = axes[cluster, 1]

            cluster_data = self.data_with_clusters[
                self.data_with_clusters["Cluster"] == cluster
            ][categories]
            cluster_data_melted = pd.melt(
                cluster_data, var_name="Região", value_name="Espessura"
            )

            sns.boxplot(
                data=cluster_data_melted,
                x="Região",
                y="Espessura",
                color=colors[cluster],
                ax=ax2,
            )
            sns.swarmplot(
                data=cluster_data_melted,
                x="Região",
                y="Espessura",
                color="black",
                alpha=0.3,
                size=2,
                ax=ax2,
            )

            ax2.set_xlabel("Região do Olho", fontweight="bold")
            ax2.set_ylabel("Espessura (μm)", fontweight="bold")
            ax2.set_title(
                f"Perfil {cluster+1}: Distribuição de Valores", fontweight="bold"
            )
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Salva
        save_path = Path(__file__).parent / "results" / "02_detailed_profiles.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Perfis detalhados salvos em: {save_path}")

        plt.show()

    def create_clinical_interpretation(self):
        """
        Cria visualização com foco em interpretação clínica
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "INTERPRETAÇÃO CLÍNICA DOS PERFIS\nCaracterísticas Distintivas",
            fontsize=16,
            fontweight="bold",
        )

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]

        # 1. Espessura Central vs Periférica
        ax1 = axes[0, 0]
        for cluster in range(self.n_clusters):
            central = self.cluster_stats.loc[cluster, ("C", "mean")]
            peripheral_regions = ["S", "ST", "T", "IT", "I", "IN", "N", "SN"]
            peripheral_mean = np.mean(
                [
                    self.cluster_stats.loc[cluster, (reg, "mean")]
                    for reg in peripheral_regions
                ]
            )

            ax1.scatter(
                central,
                peripheral_mean,
                s=self.cluster_counts[cluster] * 3,
                alpha=0.6,
                color=colors[cluster],
                label=f"Perfil {cluster+1}",
                edgecolors="black",
                linewidth=2,
            )

        ax1.plot(
            [40, 70], [40, 70], "k--", alpha=0.3, label="Linha de Igualdade"
        )  # Linha de referência
        ax1.set_xlabel("Espessura Central (μm)", fontweight="bold")
        ax1.set_ylabel("Espessura Periférica Média (μm)", fontweight="bold")
        ax1.set_title("Central vs Periférica", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Variabilidade por Perfil
        ax2 = axes[0, 1]
        variability = []
        for cluster in range(self.n_clusters):
            stds = [
                self.cluster_stats.loc[cluster, (cat, "std")]
                for cat in ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
            ]
            variability.append(np.mean(stds))

        ax2.bar(
            [f"Perfil {i+1}" for i in range(self.n_clusters)],
            variability,
            color=colors[: self.n_clusters],
            alpha=0.7,
        )
        ax2.set_ylabel("Desvio Padrão Médio (μm)", fontweight="bold")
        ax2.set_title("Variabilidade Interna dos Perfis", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. Assimetria Superior-Inferior
        ax3 = axes[1, 0]
        for cluster in range(self.n_clusters):
            superior_regions = ["S", "ST", "SN"]
            inferior_regions = ["I", "IT", "IN"]

            superior_mean = np.mean(
                [
                    self.cluster_stats.loc[cluster, (reg, "mean")]
                    for reg in superior_regions
                ]
            )
            inferior_mean = np.mean(
                [
                    self.cluster_stats.loc[cluster, (reg, "mean")]
                    for reg in inferior_regions
                ]
            )

            ax3.scatter(
                superior_mean,
                inferior_mean,
                s=self.cluster_counts[cluster] * 3,
                alpha=0.6,
                color=colors[cluster],
                label=f"Perfil {cluster+1}",
                edgecolors="black",
                linewidth=2,
            )

        ax3.plot([40, 70], [40, 70], "k--", alpha=0.3, label="Linha de Igualdade")
        ax3.set_xlabel("Espessura Superior Média (μm)", fontweight="bold")
        ax3.set_ylabel("Espessura Inferior Média (μm)", fontweight="bold")
        ax3.set_title("Assimetria Superior-Inferior", fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Tabela de Características
        ax4 = axes[1, 1]
        ax4.axis("tight")
        ax4.axis("off")

        # Prepara dados da tabela
        table_data = [["Perfil", "N", "Espessura\nMédia", "Região Mais\nEspessa"]]

        for cluster in range(self.n_clusters):
            n = self.cluster_counts[cluster]
            regions = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
            means = [self.cluster_stats.loc[cluster, (reg, "mean")] for reg in regions]
            overall_mean = np.mean(means)
            thickest_region = regions[np.argmax(means)]

            table_data.append(
                [
                    f"Perfil {cluster+1}",
                    str(n),
                    f"{overall_mean:.1f} μm",
                    thickest_region,
                ]
            )

        table = ax4.table(
            cellText=table_data,
            cellLoc="center",
            loc="center",
            colWidths=[0.2, 0.15, 0.25, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Estiliza cabeçalho
        for i in range(4):
            table[(0, i)].set_facecolor("#4ECDC4")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Colore linhas
        for i in range(1, len(table_data)):
            for j in range(4):
                table[(i, j)].set_facecolor(colors[i - 1])
                table[(i, j)].set_alpha(0.3)

        ax4.set_title(
            "Resumo das Características dos Perfis", fontweight="bold", pad=20
        )

        plt.tight_layout()

        # Salva
        save_path = Path(__file__).parent / "results" / "03_clinical_interpretation.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Interpretação clínica salva em: {save_path}")

        plt.show()

    def generate_full_report(self):
        """
        Gera relatório completo
        """
        print("\n" + "=" * 70)
        print("GERANDO APRESENTAÇÃO COMPLETA PARA O CLIENTE")
        print("=" * 70)

        self.create_executive_summary()
        self.create_detailed_profiles()
        self.create_clinical_interpretation()

        print("\n" + "=" * 70)
        print("✅ APRESENTAÇÃO COMPLETA GERADA!")
        print("=" * 70)
        print("\nArquivos salvos em: results/")
        print("  - 01_executive_summary.png")
        print("  - 02_detailed_profiles.png")
        print("  - 03_clinical_interpretation.png")


if __name__ == "__main__":
    print(
        "Este é um módulo de classes. Use o notebook 'analise_clustering.ipynb' para executar a análise."
    )
    print("Ou importe a classe ClientPresentation em seus scripts.")
