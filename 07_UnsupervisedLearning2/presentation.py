"""
Módulo de Apresentação de Resultados
Gera visualizações profissionais para apresentação ao cliente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore")


class ClientPresentation:
    """
    Classe para gerar visualizações profissionais dos resultados de clusterização.
    """

    def __init__(self):
        """Inicializa a classe de apresentação."""
        self.colors = sns.color_palette("husl", n_colors=10)
        self.features = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
        self.feature_names = {
            "C": "Central",
            "S": "Superior",
            "ST": "Sup. Temporal",
            "T": "Temporal",
            "IT": "Inf. Temporal",
            "I": "Inferior",
            "IN": "Inf. Nasal",
            "N": "Nasal",
            "SN": "Sup. Nasal",
        }

    def plot_cluster_profiles(self, df_result, save_path="results/kmeans_profiles.png"):
        """
        Gera gráfico de perfis médios de cada cluster.

        Args:
            df_result (pd.DataFrame): DataFrame com os resultados
            save_path (str): Caminho para salvar o gráfico
        """
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        n_clusters = df_result["Cluster"].nunique()

        fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 5))
        if n_clusters == 1:
            axes = [axes]

        fig.suptitle(
            "Perfis Médios de Espessura Epitelial por Cluster",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        for cluster_id in range(n_clusters):
            ax = axes[cluster_id]
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]

            # Calcular médias e desvios padrão
            means = cluster_data.mean()
            stds = cluster_data.std()

            # Criar gráfico de radar
            angles = np.linspace(
                0, 2 * np.pi, len(self.features), endpoint=False
            ).tolist()
            means_list = means.tolist()
            means_list += means_list[:1]
            angles += angles[:1]

            ax = plt.subplot(1, n_clusters, cluster_id + 1, projection="polar")

            # Plotar o perfil
            ax.plot(
                angles,
                means_list,
                "o-",
                linewidth=2,
                color=self.colors[cluster_id],
                label=f"Cluster {cluster_id}",
                markersize=8,
            )
            ax.fill(angles, means_list, alpha=0.25, color=self.colors[cluster_id])

            # Configurações
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([self.feature_names[f] for f in self.features], size=9)
            ax.set_ylim(0, 80)
            ax.set_yticks([20, 40, 60, 80])
            ax.set_yticklabels(["20", "40", "60", "80 μm"], size=8)
            ax.grid(True, alpha=0.3)

            # Título com informações
            n_olhos = len(cluster_data)
            mean_thickness = means.mean()
            ax.set_title(
                f"Cluster {cluster_id}\n{n_olhos} olhos | Média Geral: {mean_thickness:.1f} μm",
                fontsize=12,
                fontweight="bold",
                pad=20,
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico de perfis salvo em: {save_path}")
        plt.close()

    def plot_cluster_distributions(
        self, df_result, save_path="results/kmeans_distributions.png"
    ):
        """
        Gera gráfico de distribuições das espessuras por cluster.

        Args:
            df_result (pd.DataFrame): DataFrame com os resultados
            save_path (str): Caminho para salvar o gráfico
        """
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        n_clusters = df_result["Cluster"].nunique()

        fig, axes = plt.subplots(n_clusters, 3, figsize=(18, 5 * n_clusters))
        if n_clusters == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            "Distribuições de Espessura Epitelial por Cluster",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        for cluster_id in range(n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]

            # 1. Boxplot
            ax1 = axes[cluster_id, 0]
            bp = ax1.boxplot(
                [cluster_data[f].values for f in self.features],
                labels=[self.feature_names[f] for f in self.features],
                patch_artist=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(self.colors[cluster_id])
                patch.set_alpha(0.6)
            ax1.set_ylabel("Espessura (μm)", fontsize=11)
            ax1.set_title(
                f"Cluster {cluster_id} - Boxplot", fontsize=12, fontweight="bold"
            )
            ax1.grid(True, alpha=0.3, axis="y")
            ax1.tick_params(axis="x", rotation=45)

            # 2. Heatmap
            ax2 = axes[cluster_id, 1]
            corr = cluster_data.corr()
            im = ax2.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
            ax2.set_xticks(range(len(self.features)))
            ax2.set_yticks(range(len(self.features)))
            ax2.set_xticklabels(
                [self.feature_names[f] for f in self.features], rotation=45, ha="right"
            )
            ax2.set_yticklabels([self.feature_names[f] for f in self.features])
            ax2.set_title(
                f"Cluster {cluster_id} - Correlação", fontsize=12, fontweight="bold"
            )
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

            # 3. Histograma da média geral
            ax3 = axes[cluster_id, 2]
            mean_thickness = cluster_data.mean(axis=1)
            ax3.hist(
                mean_thickness,
                bins=20,
                color=self.colors[cluster_id],
                alpha=0.7,
                edgecolor="black",
            )
            ax3.axvline(
                mean_thickness.mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Média: {mean_thickness.mean():.1f} μm",
            )
            ax3.set_xlabel("Espessura Média (μm)", fontsize=11)
            ax3.set_ylabel("Frequência", fontsize=11)
            ax3.set_title(
                f"Cluster {cluster_id} - Distribuição da Média",
                fontsize=12,
                fontweight="bold",
            )
            ax3.grid(True, alpha=0.3, axis="y")
            ax3.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico de distribuições salvo em: {save_path}")
        plt.close()

    def plot_executive_summary(
        self, df_result, df_cluster_info, save_path="results/01_executive_summary.png"
    ):
        """
        Gera resumo executivo com as principais informações.

        Args:
            df_result (pd.DataFrame): DataFrame com os resultados
            df_cluster_info (pd.DataFrame): DataFrame com informações dos clusters
            save_path (str): Caminho para salvar o gráfico
        """
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        n_clusters = df_result["Cluster"].nunique()

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Título
        fig.suptitle(
            "RESUMO EXECUTIVO - Análise de Perfis de Espessura Epitelial",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # 1. Distribuição dos Clusters (Pizza)
        ax1 = fig.add_subplot(gs[0, 0])
        sizes = [len(df_result[df_result["Cluster"] == i]) for i in range(n_clusters)]
        labels = [f"Cluster {i}\n({sizes[i]} olhos)" for i in range(n_clusters)]
        ax1.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=self.colors[:n_clusters],
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax1.set_title("Distribuição dos Olhos", fontsize=13, fontweight="bold", pad=10)

        # 2. Radar Chart Comparativo
        ax2 = fig.add_subplot(gs[0, 1:], projection="polar")
        angles = np.linspace(0, 2 * np.pi, len(self.features), endpoint=False).tolist()
        angles += angles[:1]

        for cluster_id in range(n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]
            means = cluster_data.mean().tolist()
            means += means[:1]
            ax2.plot(
                angles,
                means,
                "o-",
                linewidth=2,
                color=self.colors[cluster_id],
                label=f"Cluster {cluster_id}",
                markersize=6,
            )
            ax2.fill(angles, means, alpha=0.15, color=self.colors[cluster_id])

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([self.feature_names[f] for f in self.features], size=10)
        ax2.set_ylim(0, 80)
        ax2.set_title(
            "Comparação dos Perfis Médios", fontsize=13, fontweight="bold", pad=20
        )
        ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax2.grid(True, alpha=0.3)

        # 3. Tabela de Características
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis("tight")
        ax3.axis("off")

        # Preparar dados da tabela
        table_data = []
        for cluster_id in range(n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]
            row = [
                f"Cluster {cluster_id}",
                f"{len(cluster_data)}",
                f"{cluster_data.mean().mean():.1f} ± {cluster_data.mean().std():.1f}",
                f'{cluster_data["C"].mean():.1f} ± {cluster_data["C"].std():.1f}',
                f'{cluster_data[["S", "I"]].mean().mean():.1f}',
                f'{cluster_data[["T", "N"]].mean().mean():.1f}',
                f"{cluster_data.std().mean():.1f}",
            ]
            table_data.append(row)

        columns = [
            "Cluster",
            "N° Olhos",
            "Média Geral (μm)",
            "Central (μm)",
            "S-I Médio (μm)",
            "T-N Médio (μm)",
            "Variabilidade",
        ]

        table = ax3.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Estilizar cabeçalho
        for i in range(len(columns)):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Colorir linhas dos clusters
        for i in range(n_clusters):
            for j in range(len(columns)):
                table[(i + 1, j)].set_facecolor(self.colors[i])
                table[(i + 1, j)].set_alpha(0.3)

        ax3.set_title(
            "Características Principais dos Clusters",
            fontsize=13,
            fontweight="bold",
            pad=20,
        )

        # 4. Comparação de Médias por Região
        ax4 = fig.add_subplot(gs[2, :])

        x = np.arange(len(self.features))
        width = 0.8 / n_clusters

        for cluster_id in range(n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]
            means = cluster_data.mean()
            offset = width * (cluster_id - n_clusters / 2 + 0.5)
            ax4.bar(
                x + offset,
                means,
                width,
                label=f"Cluster {cluster_id}",
                color=self.colors[cluster_id],
                alpha=0.8,
            )

        ax4.set_xlabel("Região", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Espessura Média (μm)", fontsize=12, fontweight="bold")
        ax4.set_title(
            "Comparação de Espessura por Região", fontsize=13, fontweight="bold"
        )
        ax4.set_xticks(x)
        ax4.set_xticklabels([self.feature_names[f] for f in self.features])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Resumo executivo salvo em: {save_path}")
        plt.close()

    def plot_detailed_profiles(
        self, df_result, save_path="results/02_detailed_profiles.png"
    ):
        """
        Gera análise detalhada de cada cluster individualmente.

        Args:
            df_result (pd.DataFrame): DataFrame com os resultados
            save_path (str): Caminho para salvar o gráfico
        """
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        n_clusters = df_result["Cluster"].nunique()

        fig, axes = plt.subplots(n_clusters, 4, figsize=(20, 5 * n_clusters))
        if n_clusters == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            "ANÁLISE DETALHADA POR CLUSTER", fontsize=18, fontweight="bold", y=0.995
        )

        for cluster_id in range(n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]

            # 1. Perfil Espacial (Mapa de Calor)
            ax1 = axes[cluster_id, 0]
            spatial_map = np.array(
                [
                    [
                        cluster_data["ST"].mean(),
                        cluster_data["S"].mean(),
                        cluster_data["SN"].mean(),
                    ],
                    [
                        cluster_data["T"].mean(),
                        cluster_data["C"].mean(),
                        cluster_data["N"].mean(),
                    ],
                    [
                        cluster_data["IT"].mean(),
                        cluster_data["I"].mean(),
                        cluster_data["IN"].mean(),
                    ],
                ]
            )

            im = ax1.imshow(spatial_map, cmap="YlOrRd", aspect="auto", vmin=40, vmax=70)
            ax1.set_xticks([0, 1, 2])
            ax1.set_yticks([0, 1, 2])
            ax1.set_xticklabels(["Temporal", "Central", "Nasal"])
            ax1.set_yticklabels(["Superior", "Médio", "Inferior"])

            # Adicionar valores
            for i in range(3):
                for j in range(3):
                    text = ax1.text(
                        j,
                        i,
                        f"{spatial_map[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                    )

            ax1.set_title(
                f"Cluster {cluster_id} - Mapa Espacial\n({len(cluster_data)} olhos)",
                fontsize=12,
                fontweight="bold",
            )
            plt.colorbar(im, ax=ax1, label="Espessura (μm)")

            # 2. Boxplot Comparativo
            ax2 = axes[cluster_id, 1]
            data_to_plot = [cluster_data[f].values for f in self.features]
            bp = ax2.boxplot(
                data_to_plot,
                labels=[self.feature_names[f] for f in self.features],
                patch_artist=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(self.colors[cluster_id])
                patch.set_alpha(0.6)
            ax2.set_ylabel("Espessura (μm)", fontsize=11)
            ax2.set_title(
                f"Cluster {cluster_id} - Variação por Região",
                fontsize=12,
                fontweight="bold",
            )
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(True, alpha=0.3, axis="y")

            # 3. Radar Chart
            ax3 = axes[cluster_id, 2]
            ax3 = plt.subplot(n_clusters, 4, cluster_id * 4 + 3, projection="polar")

            angles = np.linspace(
                0, 2 * np.pi, len(self.features), endpoint=False
            ).tolist()
            means = cluster_data.mean().tolist()
            stds = cluster_data.std().tolist()
            angles += angles[:1]
            means += means[:1]

            ax3.plot(
                angles,
                means,
                "o-",
                linewidth=2,
                color=self.colors[cluster_id],
                markersize=8,
            )
            ax3.fill(angles, means, alpha=0.25, color=self.colors[cluster_id])

            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels([self.feature_names[f] for f in self.features], size=9)
            ax3.set_ylim(0, 80)
            ax3.set_title(
                f"Cluster {cluster_id} - Perfil Radial",
                fontsize=12,
                fontweight="bold",
                pad=20,
            )
            ax3.grid(True, alpha=0.3)

            # 4. Estatísticas
            ax4 = axes[cluster_id, 3]
            ax4.axis("off")

            stats_text = f"""
CLUSTER {cluster_id}
{'=' * 30}

Tamanho: {len(cluster_data)} olhos

MÉDIAS (μm):
  Central:     {cluster_data['C'].mean():.1f} ± {cluster_data['C'].std():.1f}
  Superior:    {cluster_data['S'].mean():.1f} ± {cluster_data['S'].std():.1f}
  Inferior:    {cluster_data['I'].mean():.1f} ± {cluster_data['I'].std():.1f}
  Temporal:    {cluster_data['T'].mean():.1f} ± {cluster_data['T'].std():.1f}
  Nasal:       {cluster_data['N'].mean():.1f} ± {cluster_data['N'].std():.1f}

ASSIMETRIAS:
  Superior-Inferior: {cluster_data['S'].mean() - cluster_data['I'].mean():+.1f} μm
  Temporal-Nasal:    {cluster_data['T'].mean() - cluster_data['N'].mean():+.1f} μm

VARIABILIDADE:
  Média do Desvio:   {cluster_data.std().mean():.1f} μm
  Mínima:            {cluster_data.std().min():.1f} μm
  Máxima:            {cluster_data.std().max():.1f} μm
            """

            ax4.text(
                0.1,
                0.95,
                stats_text,
                transform=ax4.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round", facecolor=self.colors[cluster_id], alpha=0.2
                ),
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Análise detalhada salva em: {save_path}")
        plt.close()

    def plot_clinical_interpretation(
        self, df_result, save_path="results/03_clinical_interpretation.png"
    ):
        """
        Gera interpretação clínica dos clusters.

        Args:
            df_result (pd.DataFrame): DataFrame com os resultados
            save_path (str): Caminho para salvar o gráfico
        """
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        n_clusters = df_result["Cluster"].nunique()

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

        fig.suptitle(
            "INTERPRETAÇÃO CLÍNICA DOS PERFIS DE ESPESSURA EPITELIAL",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # 1. Assimetrias Superior-Inferior
        ax1 = fig.add_subplot(gs[0, 0])
        si_asym = []
        for cluster_id in range(n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]
            asym = cluster_data["S"].mean() - cluster_data["I"].mean()
            si_asym.append(asym)

        bars = ax1.bar(
            range(n_clusters), si_asym, color=self.colors[:n_clusters], alpha=0.8
        )
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax1.set_xlabel("Cluster", fontsize=12, fontweight="bold")
        ax1.set_ylabel(
            "Diferença Superior - Inferior (μm)", fontsize=12, fontweight="bold"
        )
        ax1.set_title("Assimetria Superior-Inferior", fontsize=13, fontweight="bold")
        ax1.set_xticks(range(n_clusters))
        ax1.set_xticklabels([f"Cluster {i}" for i in range(n_clusters)])
        ax1.grid(True, alpha=0.3, axis="y")

        # Adicionar valores
        for i, (bar, val) in enumerate(zip(bars, si_asym)):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:+.1f}",
                ha="center",
                va="bottom" if val > 0 else "top",
                fontweight="bold",
            )

        # 2. Assimetrias Temporal-Nasal
        ax2 = fig.add_subplot(gs[0, 1])
        tn_asym = []
        for cluster_id in range(n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]
            asym = cluster_data["T"].mean() - cluster_data["N"].mean()
            tn_asym.append(asym)

        bars = ax2.bar(
            range(n_clusters), tn_asym, color=self.colors[:n_clusters], alpha=0.8
        )
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax2.set_xlabel("Cluster", fontsize=12, fontweight="bold")
        ax2.set_ylabel(
            "Diferença Temporal - Nasal (μm)", fontsize=12, fontweight="bold"
        )
        ax2.set_title("Assimetria Temporal-Nasal", fontsize=13, fontweight="bold")
        ax2.set_xticks(range(n_clusters))
        ax2.set_xticklabels([f"Cluster {i}" for i in range(n_clusters)])
        ax2.grid(True, alpha=0.3, axis="y")

        for i, (bar, val) in enumerate(zip(bars, tn_asym)):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:+.1f}",
                ha="center",
                va="bottom" if val > 0 else "top",
                fontweight="bold",
            )

        # 3. Variabilidade Intra-Cluster
        ax3 = fig.add_subplot(gs[1, 0])
        variabilities = []
        for cluster_id in range(n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]
            var = cluster_data.std().mean()
            variabilities.append(var)

        bars = ax3.bar(
            range(n_clusters), variabilities, color=self.colors[:n_clusters], alpha=0.8
        )
        ax3.set_xlabel("Cluster", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Desvio Padrão Médio (μm)", fontsize=12, fontweight="bold")
        ax3.set_title("Homogeneidade dos Clusters", fontsize=13, fontweight="bold")
        ax3.set_xticks(range(n_clusters))
        ax3.set_xticklabels([f"Cluster {i}" for i in range(n_clusters)])
        ax3.grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Relação Central vs Periférica
        ax4 = fig.add_subplot(gs[1, 1])
        for cluster_id in range(n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]
            central = cluster_data["C"].values
            peripheral = (
                cluster_data[["S", "ST", "T", "IT", "I", "IN", "N", "SN"]]
                .mean(axis=1)
                .values
            )
            ax4.scatter(
                central,
                peripheral,
                color=self.colors[cluster_id],
                label=f"Cluster {cluster_id}",
                alpha=0.6,
                s=50,
            )

        # Linha de referência (central = periférica)
        min_val = df_result["C"].min()
        max_val = df_result["C"].max()
        ax4.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            alpha=0.5,
            label="Central = Periférica",
        )

        ax4.set_xlabel("Espessura Central (μm)", fontsize=12, fontweight="bold")
        ax4.set_ylabel(
            "Espessura Periférica Média (μm)", fontsize=12, fontweight="bold"
        )
        ax4.set_title("Relação Central vs Periférica", fontsize=13, fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Interpretação Textual
        ax5 = fig.add_subplot(gs[2:, :])
        ax5.axis("off")

        interpretation_text = (
            "INTERPRETAÇÃO CLÍNICA DOS CLUSTERS\n" + "=" * 120 + "\n\n"
        )

        for cluster_id in range(n_clusters):
            cluster_data = df_result[df_result["Cluster"] == cluster_id][self.features]

            # Cálculos
            mean_overall = cluster_data.mean().mean()
            mean_central = cluster_data["C"].mean()
            mean_superior = cluster_data["S"].mean()
            mean_inferior = cluster_data["I"].mean()
            mean_temporal = cluster_data["T"].mean()
            mean_nasal = cluster_data["N"].mean()
            si_diff = mean_superior - mean_inferior
            tn_diff = mean_temporal - mean_nasal
            variability = cluster_data.std().mean()

            interpretation_text += f"CLUSTER {cluster_id} ({len(cluster_data)} olhos - {len(cluster_data)/len(df_result)*100:.1f}%):\n"
            interpretation_text += "-" * 120 + "\n"

            # Espessura geral
            if mean_overall < 50:
                interpretation_text += f"• Espessura FINA (média: {mean_overall:.1f} μm) - Pode indicar epitélio fino ou atrofia.\n"
            elif mean_overall > 60:
                interpretation_text += f"• Espessura GROSSA (média: {mean_overall:.1f} μm) - Pode indicar epitélio espesso ou edema.\n"
            else:
                interpretation_text += f"• Espessura NORMAL (média: {mean_overall:.1f} μm) - Dentro do padrão esperado.\n"

            # Central vs periférica
            periph_mean = (
                cluster_data[["S", "ST", "T", "IT", "I", "IN", "N", "SN"]].mean().mean()
            )
            if mean_central > periph_mean + 2:
                interpretation_text += f"• Padrão CENTRAL DOMINANTE (C: {mean_central:.1f} vs P: {periph_mean:.1f} μm).\n"
            elif mean_central < periph_mean - 2:
                interpretation_text += f"• Padrão PERIFÉRICO DOMINANTE (C: {mean_central:.1f} vs P: {periph_mean:.1f} μm).\n"
            else:
                interpretation_text += f"• Padrão UNIFORME (C: {mean_central:.1f} vs P: {periph_mean:.1f} μm).\n"

            # Assimetrias
            if abs(si_diff) > 3:
                direction = "SUPERIOR" if si_diff > 0 else "INFERIOR"
                interpretation_text += f"• Assimetria S-I SIGNIFICATIVA: predominância {direction} ({si_diff:+.1f} μm).\n"
            else:
                interpretation_text += (
                    f"• Simetria S-I PRESERVADA ({si_diff:+.1f} μm).\n"
                )

            if abs(tn_diff) > 3:
                direction = "TEMPORAL" if tn_diff > 0 else "NASAL"
                interpretation_text += f"• Assimetria T-N SIGNIFICATIVA: predominância {direction} ({tn_diff:+.1f} μm).\n"
            else:
                interpretation_text += (
                    f"• Simetria T-N PRESERVADA ({tn_diff:+.1f} μm).\n"
                )

            # Variabilidade
            if variability < 4:
                interpretation_text += f"• HOMOGÊNEO: baixa variabilidade intra-cluster (DP médio: {variability:.1f} μm).\n"
            elif variability > 6:
                interpretation_text += f"• HETEROGÊNEO: alta variabilidade intra-cluster (DP médio: {variability:.1f} μm).\n"
            else:
                interpretation_text += f"• MODERADAMENTE UNIFORME: variabilidade média (DP médio: {variability:.1f} μm).\n"

            interpretation_text += "\n"

        ax5.text(
            0.05,
            0.95,
            interpretation_text,
            transform=ax5.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Interpretação clínica salva em: {save_path}")
        plt.close()

    def generate_all(self, df_result, df_cluster_info):
        """
        Gera todas as visualizações de uma vez.

        Args:
            df_result (pd.DataFrame): DataFrame com os resultados
            df_cluster_info (pd.DataFrame): DataFrame com informações dos clusters
        """
        print("\n" + "=" * 60)
        print("GERANDO VISUALIZAÇÕES PARA APRESENTAÇÃO")
        print("=" * 60)

        self.plot_cluster_profiles(df_result)
        self.plot_cluster_distributions(df_result)
        self.plot_executive_summary(df_result, df_cluster_info)
        self.plot_detailed_profiles(df_result)
        self.plot_clinical_interpretation(df_result)

        print("\n✓ Todas as visualizações foram geradas com sucesso!")


if __name__ == "__main__":
    # Exemplo de uso
    df_result = pd.read_csv("results/kmeans_results.csv")

    presentation = ClientPresentation()
    presentation.generate_all(df_result, None)
