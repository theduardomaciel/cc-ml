import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
from pathlib import Path

# Adicionar o diretório pai ao sys.path para permitir importação
sys.path.append(str(Path(__file__).parent.parent))
from kmeans_clustering import KMeansClusterer

warnings.filterwarnings("ignore")

# Configurar estilo para apresentação profissional
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ClientPresentation:
    def __init__(self):
        """
        Classe para gerar visualizações profissionais dos clusters para apresentação ao cliente
        """
        self.clusterer = None
        self.data = None
        self.cluster_stats = None
        self.cluster_counts = None

    def setup_data(self):
        """
        Configura e processa os dados
        """
        # Configurações
        data_path = (
            Path(__file__).parent.parent / "data" / "barrettII_eyes_clustering.csv"
        )
        n_clusters = 3

        # Inicializa o clusterer
        self.clusterer = KMeansClusterer(n_clusters=n_clusters)

        # Carrega e preprocessa os dados
        self.clusterer.load_data(data_path)
        self.clusterer.preprocess_data()

        # Treina o modelo
        self.clusterer.fit()

        # Obtém perfis dos clusters
        self.cluster_stats, self.cluster_counts = self.clusterer.get_cluster_profiles()

        # Dados com clusters
        self.data = self.clusterer.features_data.copy()
        self.data["Cluster"] = self.clusterer.labels

    def create_executive_summary(self, save_path=None):
        """
        Cria um resumo executivo dos clusters encontrados
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "PERFIL DOS GRUPOS DE PACIENTES OFTALMOLÓGICOS\nResumo Executivo",
            fontsize=18,
            fontweight="bold",
            y=0.95,
        )

        # 1. Distribuição dos pacientes por grupo
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        wedges, texts, autotexts = ax1.pie(
            self.cluster_counts.values,
            labels=[f"Grupo {i+1}" for i in range(len(self.cluster_counts))],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            explode=(0.05, 0.05, 0.05),
        )

        ax1.set_title(
            "Distribuição de Pacientes por Grupo",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Adicionar contagem absoluta
        for i, (count, autotext) in enumerate(
            zip(self.cluster_counts.values, autotexts)
        ):
            autotext.set_text(f"{autotext.get_text()}\n({count} pacientes)")
            autotext.set_fontsize(10)
            autotext.set_fontweight("bold")

        # 2. Características médias por cluster - Radar Chart
        categories = ["AL (mm)", "ACD (mm)", "WTW (mm)", "K1 (D)", "K2 (D)"]

        # Normalizar dados para radar chart (0-1)
        means_data = []
        for cluster in range(3):
            cluster_means = []
            for feature in ["AL", "ACD", "WTW", "K1", "K2"]:
                value = self.cluster_stats.loc[cluster, (feature, "mean")]
                cluster_means.append(value)
            means_data.append(cluster_means)

        # Normalizar para escala 0-1
        all_values = np.array(means_data)
        min_vals = all_values.min(axis=0)
        max_vals = all_values.max(axis=0)
        normalized_data = (all_values - min_vals) / (max_vals - min_vals)

        # Criar radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Fechar o círculo

        ax2 = plt.subplot(2, 2, 2, projection="polar")

        for i, (cluster_data, color) in enumerate(zip(normalized_data, colors)):
            values = cluster_data.tolist()
            values += values[:1]  # Fechar o círculo
            ax2.plot(
                angles, values, "o-", linewidth=2, label=f"Grupo {i+1}", color=color
            )
            ax2.fill(angles, values, alpha=0.25, color=color)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 1)
        ax2.set_title(
            "Perfil Comparativo dos Grupos\n(Valores Normalizados)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        # 3. Boxplot das principais características
        features_to_plot = ["AL", "K1"]
        data_melted = pd.melt(
            self.data[features_to_plot + ["Cluster"]],
            id_vars=["Cluster"],
            var_name="Característica",
            value_name="Valor",
        )

        sns.boxplot(
            data=data_melted,
            x="Característica",
            y="Valor",
            hue="Cluster",
            palette=colors,
            ax=ax3,
        )
        ax3.set_title(
            "Distribuição das Principais Características",
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_xlabel("Medidas Oculares", fontsize=12)
        ax3.set_ylabel("Valores", fontsize=12)
        ax3.legend(title="Grupo", labels=["Grupo 1", "Grupo 2", "Grupo 3"])

        # 4. Tabela resumo
        ax4.axis("tight")
        ax4.axis("off")

        # Criar tabela resumo
        summary_data = []
        for cluster in range(3):
            row = [
                f"Grupo {cluster + 1}",
                f"{self.cluster_counts[cluster]} ({self.cluster_counts[cluster]/self.cluster_counts.sum()*100:.1f}%)",
                f'{self.cluster_stats.loc[cluster, ("AL", "mean")]:.2f} ± {self.cluster_stats.loc[cluster, ("AL", "std")]:.2f}',
                f'{self.cluster_stats.loc[cluster, ("K1", "mean")]:.2f} ± {self.cluster_stats.loc[cluster, ("K1", "std")]:.2f}',
                f'{self.cluster_stats.loc[cluster, ("K2", "mean")]:.2f} ± {self.cluster_stats.loc[cluster, ("K2", "std")]:.2f}',
            ]
            summary_data.append(row)

        table = ax4.table(
            cellText=summary_data,
            colLabels=[
                "Grupo",
                "Pacientes",
                "Comprimento Axial\n(AL mm)",
                "Curvatura K1\n(Dioptrias)",
                "Curvatura K2\n(Dioptrias)",
            ],
            cellLoc="center",
            loc="center",
            colColours=["lightblue"] * 5,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Colorir as linhas por grupo
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                table[(i + 1, j)].set_facecolor(colors[i])
                table[(i + 1, j)].set_alpha(0.3)

        ax4.set_title(
            "Resumo Estatístico por Grupo", fontsize=14, fontweight="bold", pad=20
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def create_detailed_profiles(self, save_path=None):
        """
        Cria perfis detalhados de cada grupo
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(
            "PERFIS DETALHADOS DOS GRUPOS DE PACIENTES",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        group_names = [
            "Grupo 1: Olhos Curtos",
            "Grupo 2: Olhos Longos",
            "Grupo 3: Olhos Normais",
        ]

        for cluster in range(3):
            # Dados do cluster atual
            cluster_data = self.data[self.data["Cluster"] == cluster]

            # Gráfico de distribuição das características
            ax1 = axes[cluster, 0]
            features = ["AL", "ACD", "WTW", "K1", "K2"]
            means = [
                self.cluster_stats.loc[cluster, (feature, "mean")]
                for feature in features
            ]
            stds = [
                self.cluster_stats.loc[cluster, (feature, "std")]
                for feature in features
            ]

            x_pos = np.arange(len(features))
            bars = ax1.bar(
                x_pos,
                means,
                yerr=stds,
                capsize=5,
                color=colors[cluster],
                alpha=0.7,
                edgecolor="black",
            )

            ax1.set_xlabel("Características Oculares")
            ax1.set_ylabel("Valores")
            ax1.set_title(
                f"{group_names[cluster]}\n({self.cluster_counts[cluster]} pacientes)",
                fontweight="bold",
            )
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(features)
            ax1.grid(True, alpha=0.3)

            # Adicionar valores nas barras
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + std + 0.5,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Scatter plot 2D
            ax2 = axes[cluster, 1]
            scatter = ax2.scatter(
                cluster_data["AL"],
                cluster_data["K1"],
                c=colors[cluster],
                alpha=0.6,
                s=30,
            )
            ax2.set_xlabel("Comprimento Axial (AL) - mm")
            ax2.set_ylabel("Curvatura Principal (K1) - Dioptrias")
            ax2.set_title(f'Dispersão AL vs K1 - {group_names[cluster].split(":")[0]}')
            ax2.grid(True, alpha=0.3)

            # Adicionar linha de tendência
            z = np.polyfit(cluster_data["AL"], cluster_data["K1"], 1)
            p = np.poly1d(z)
            ax2.plot(
                cluster_data["AL"], p(cluster_data["AL"]), "--", color="red", alpha=0.8
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def create_clinical_interpretation(self, save_path=None):
        """
        Cria interpretação clínica dos grupos
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "INTERPRETAÇÃO CLÍNICA DOS GRUPOS IDENTIFICADOS",
            fontsize=18,
            fontweight="bold",
            y=0.95,
        )

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # 1. Relação AL vs K1 (importante clinicamente)
        for cluster in range(3):
            cluster_data = self.data[self.data["Cluster"] == cluster]
            ax1.scatter(
                cluster_data["AL"],
                cluster_data["K1"],
                c=colors[cluster],
                label=f"Grupo {cluster+1}",
                alpha=0.6,
                s=30,
            )

        ax1.set_xlabel("Comprimento Axial (mm)")
        ax1.set_ylabel("Curvatura Corneana K1 (Dioptrias)")
        ax1.set_title(
            "Relação Comprimento-Curvatura\n(Diagnóstico de Ametropia)",
            fontweight="bold",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Distribuição da profundidade da câmara anterior
        for cluster in range(3):
            cluster_data = self.data[self.data["Cluster"] == cluster]
            ax2.hist(
                cluster_data["ACD"],
                bins=20,
                alpha=0.7,
                color=colors[cluster],
                label=f"Grupo {cluster+1}",
                density=True,
            )

        ax2.set_xlabel("Profundidade da Câmara Anterior (mm)")
        ax2.set_ylabel("Densidade de Probabilidade")
        ax2.set_title(
            "Distribuição da Profundidade\nda Câmara Anterior", fontweight="bold"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Análise de astigmatismo (K1 vs K2)
        for cluster in range(3):
            cluster_data = self.data[self.data["Cluster"] == cluster]
            astigmatism = abs(cluster_data["K1"] - cluster_data["K2"])
            ax3.scatter(
                cluster_data["K1"],
                cluster_data["K2"],
                c=astigmatism,
                s=30,
                alpha=0.7,
                cmap="YlOrRd",
            )

        ax3.set_xlabel("Curvatura Principal K1 (Dioptrias)")
        ax3.set_ylabel("Curvatura Principal K2 (Dioptrias)")
        ax3.set_title("Análise de Astigmatismo\n(Diferença K1-K2)", fontweight="bold")

        # Linha de igualdade (sem astigmatismo)
        min_k = min(self.data[["K1", "K2"]].min())
        max_k = max(self.data[["K1", "K2"]].max())
        ax3.plot(
            [min_k, max_k],
            [min_k, max_k],
            "--",
            color="black",
            alpha=0.5,
            label="Sem Astigmatismo",
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Características clínicas por grupo
        ax4.axis("tight")
        ax4.axis("off")

        clinical_interpretation = [
            [
                "Grupo 1\n(Olhos Curtos)",
                "AL < 23mm\nK > 45D",
                "Hipermetropia\nRisco: Glaucoma",
                "Lentes divergentes\nMonitoramento IOP",
            ],
            [
                "Grupo 2\n(Olhos Longos)",
                "AL > 24mm\nK < 42D",
                "Miopia\nRisco: Degeneração",
                "Lentes convergentes\nExame de retina",
            ],
            [
                "Grupo 3\n(Olhos Normais)",
                "23-24mm\n42-45D",
                "Emetropia\nBaixo risco",
                "Acompanhamento\npreventivo",
            ],
        ]

        table = ax4.table(
            cellText=clinical_interpretation,
            colLabels=["Grupo", "Características", "Condição Clínica", "Recomendações"],
            cellLoc="center",
            loc="center",
            colColours=["lightblue"] * 4,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)

        # Colorir as linhas por grupo
        for i in range(len(clinical_interpretation)):
            for j in range(len(clinical_interpretation[0])):
                table[(i + 1, j)].set_facecolor(colors[i])
                table[(i + 1, j)].set_alpha(0.3)

        ax4.set_title(
            "Interpretação e Recomendações Clínicas",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def generate_report(self):
        """
        Gera relatório completo para apresentação ao cliente
        """
        print("=" * 60)
        print("RELATÓRIO EXECUTIVO - SEGMENTAÇÃO DE PACIENTES OFTALMOLÓGICOS")
        print("=" * 60)

        print(f"\n📊 RESUMO DOS DADOS:")
        print(f"• Total de pacientes analisados: {self.cluster_counts.sum():,}")
        print(
            f"• Características analisadas: Comprimento Axial, Profundidade da Câmara, Diâmetro Corneano, Curvaturas"
        )
        print(f"• Grupos identificados: {len(self.cluster_counts)}")

        print(f"\n🎯 SEGMENTAÇÃO IDENTIFICADA:")

        group_descriptions = [
            "OLHOS CURTOS (Tendência Hipermetrópica)",
            "OLHOS LONGOS (Tendência Miopica)",
            "OLHOS PADRÃO (Emetrópicos)",
        ]

        for i, (cluster, count) in enumerate(self.cluster_counts.items()):
            percentage = (count / self.cluster_counts.sum()) * 100
            print(f"\n• GRUPO {cluster + 1} - {group_descriptions[i]}:")
            print(f"  - Pacientes: {count:,} ({percentage:.1f}%)")
            print(
                f"  - Comprimento Axial: {self.cluster_stats.loc[cluster, ('AL', 'mean')]:.2f} ± {self.cluster_stats.loc[cluster, ('AL', 'std')]:.2f} mm"
            )
            print(
                f"  - Curvatura K1: {self.cluster_stats.loc[cluster, ('K1', 'mean')]:.2f} ± {self.cluster_stats.loc[cluster, ('K1', 'std')]:.2f} D"
            )
            print(
                f"  - Curvatura K2: {self.cluster_stats.loc[cluster, ('K2', 'mean')]:.2f} ± {self.cluster_stats.loc[cluster, ('K2', 'std')]:.2f} D"
            )

        print(f"\n💡 INSIGHTS PARA O NEGÓCIO:")
        print(
            f"• Grupo mais numeroso: Grupo 3 ({self.cluster_counts[2]:,} pacientes - {(self.cluster_counts[2]/self.cluster_counts.sum())*100:.1f}%)"
        )
        print(f"• Oportunidade de especialização em correção miópica (Grupo 2)")
        print(
            f"• Necessidade de monitoramento especializado para Grupo 1 (risco de glaucoma)"
        )
        print(f"• Segmentação permite personalização de tratamentos e produtos")

        print(f"\n📈 RECOMENDAÇÕES ESTRATÉGICAS:")
        print(f"• Desenvolver protocolos específicos para cada grupo")
        print(f"• Criar campanhas de marketing segmentadas")
        print(f"• Investir em equipamentos específicos para cada perfil")
        print(f"• Treinar equipe para identificação rápida dos grupos")

        print("=" * 60)


def main():
    """
    Função principal para gerar apresentação completa
    """
    # Criar pasta para resultados
    results_path = Path(__file__).parent / "client_presentation_results"
    results_path.mkdir(exist_ok=True)

    # Inicializar apresentação
    presentation = ClientPresentation()
    presentation.setup_data()

    # Gerar relatório textual
    presentation.generate_report()

    # Gerar visualizações
    print("\nGerando visualizações...")

    # 1. Resumo executivo
    presentation.create_executive_summary(
        save_path=results_path / "01_executive_summary.png"
    )

    # 2. Perfis detalhados
    presentation.create_detailed_profiles(
        save_path=results_path / "02_detailed_profiles.png"
    )

    # 3. Interpretação clínica
    presentation.create_clinical_interpretation(
        save_path=results_path / "03_clinical_interpretation.png"
    )

    print(f"\n✅ Apresentação completa gerada em: {results_path}")
    print("\nArquivos criados:")
    print("• 01_executive_summary.png - Resumo executivo")
    print("• 02_detailed_profiles.png - Perfis detalhados dos grupos")
    print("• 03_clinical_interpretation.png - Interpretação clínica")


if __name__ == "__main__":
    main()
