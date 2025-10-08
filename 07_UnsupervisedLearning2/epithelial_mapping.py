import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Wedge, Circle
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize


def create_epithelial_map(
    data, title="Mapeamento Epitelial", save_path=None, cmap="RdYlGn_r"
):
    """
    Cria mapa epitelial circular similar aos usados em oftalmologia

    Args:
        data: dict com valores para cada região (C, S, ST, T, IT, I, IN, N, SN)
        title: título do mapa
        save_path: caminho para salvar a imagem
        cmap: colormap a usar
    """

    # Regiões em ordem circular (sentido horário a partir do topo)
    # C = Centro, S = Superior, ST = Superotemporal, T = Temporal, etc.
    regions_order = ["S", "ST", "T", "IT", "I", "IN", "N", "SN"]
    center_region = "C"

    # Ângulos para cada região (360° / 8 regiões = 45° cada)
    angles = np.linspace(90, 450, len(regions_order) + 1)  # Começa no topo

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))

    # Valores para colormap
    all_values = [data.get(r, 0) for r in regions_order] + [data.get(center_region, 0)]
    vmin, vmax = min(all_values), max(all_values)

    # Normalização para cores
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.colormaps.get_cmap(cmap)

    # Desenha regiões periféricas (anel externo)
    for i, region in enumerate(regions_order):
        value = data.get(region, 0)
        color = cmap_obj(norm(value))

        # Cria wedge (fatia de pizza)
        wedge = Wedge(
            center=(0, 0),
            r=1.0,
            theta1=angles[i],
            theta2=angles[i + 1],
            width=0.5,  # Cria anel (raio interno = 0.5)
            facecolor=color,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(wedge)

        # Adiciona label e valor
        angle_mid = np.radians((angles[i] + angles[i + 1]) / 2)
        x_text = 0.75 * np.cos(angle_mid)
        y_text = 0.75 * np.sin(angle_mid)

        ax.text(
            x_text,
            y_text,
            f"{region}\n{value:.1f}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Desenha região central
    center_value = data.get(center_region, 0)
    center_color = cmap_obj(norm(center_value))

    circle = Circle((0, 0), 0.5, facecolor=center_color, edgecolor="black", linewidth=2)
    ax.add_patch(circle)

    ax.text(
        0,
        0,
        f"{center_region}\n{center_value:.1f}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
    )

    # Configurações do plot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Adiciona colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar.set_label("Espessura Epitelial (μm)", fontsize=12, fontweight="bold")

    # Adiciona legenda de regiões
    legend_text = (
        "C: Centro | S: Superior | ST: Superotemporal | T: Temporal\n"
        "IT: Inferotemporal | I: Inferior | IN: Inferonasal | N: Nasal | SN: Superonasal"
    )
    ax.text(
        0,
        -1.4,
        legend_text,
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Mapa salvo em: {save_path}")

    return fig, ax


def create_multiple_maps(df, output_dir="results_epithelial_maps", n_samples=6):
    """
    Cria múltiplos mapas epiteliais para diferentes amostras

    Args:
        df: DataFrame com dados
        output_dir: diretório para salvar mapas
        n_samples: número de amostras aleatórias para plotar
    """

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    regions = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]

    # Seleciona amostras aleatórias
    samples = df.sample(n=min(n_samples, len(df)), random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Exemplos de Mapeamentos Epiteliais", fontsize=18, fontweight="bold")

    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= 6:
            break

        ax = axes[idx // 3, idx % 3]

        # Prepara dados para o mapa
        data = {region: row[region] for region in regions}

        # Cria mapa simplificado
        plot_simple_map(ax, data, row)

    plt.tight_layout()
    plt.savefig(
        output_path / "sample_epithelial_maps.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ {n_samples} mapas de exemplo salvos")

    # Cria mapas individuais detalhados
    for idx, (_, row) in enumerate(samples.head(3).iterrows()):
        data = {region: row[region] for region in regions}
        title = (
            f"Mapeamento Epitelial - Paciente {row.get('pID', 'N/A')}\n"
            f"Idade: {row.get('Age', 'N/A')} | Sexo: {row.get('Gender', 'N/A')} | Olho: {row.get('Eye', 'N/A')}"
        )

        create_epithelial_map(
            data,
            title=title,
            save_path=output_path / f"detailed_map_{idx+1}.png",
            cmap="RdYlGn_r",
        )


def plot_simple_map(ax, data, row):
    """Versão simplificada do mapa para subplots"""

    regions_order = ["S", "ST", "T", "IT", "I", "IN", "N", "SN"]
    center_region = "C"

    angles = np.linspace(90, 450, len(regions_order) + 1)

    all_values = [data.get(r, 0) for r in regions_order] + [data.get(center_region, 0)]
    vmin, vmax = min(all_values), max(all_values)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.colormaps.get_cmap("RdYlGn_r")

    # Regiões periféricas
    for i, region in enumerate(regions_order):
        value = data.get(region, 0)
        color = cmap_obj(norm(value))

        wedge = Wedge(
            center=(0, 0),
            r=1.0,
            theta1=angles[i],
            theta2=angles[i + 1],
            width=0.5,
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(wedge)

        # Label simplificado
        angle_mid = np.radians((angles[i] + angles[i + 1]) / 2)
        x_text = 0.75 * np.cos(angle_mid)
        y_text = 0.75 * np.sin(angle_mid)

        ax.text(
            x_text,
            y_text,
            f"{value:.0f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # Centro
    center_value = data.get(center_region, 0)
    center_color = cmap_obj(norm(center_value))

    circle = Circle((0, 0), 0.5, facecolor=center_color, edgecolor="black", linewidth=2)
    ax.add_patch(circle)

    ax.text(
        0,
        0,
        f"{center_value:.0f}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Título com info do paciente
    title = f"pID: {row.get('pID', 'N/A')} | {row.get('Age', 'N/A')}a | {row.get('Gender', 'N/A')} | {row.get('Eye', 'N/A')}"
    ax.set_title(title, fontsize=10, pad=10)


def create_cluster_average_maps(
    df, cluster_labels, output_dir="results_epithelial_maps"
):
    """
    Cria mapas epiteliais médios para cada cluster

    Args:
        df: DataFrame com dados
        cluster_labels: array com labels dos clusters
        output_dir: diretório para salvar mapas
    """

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    regions = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]

    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = cluster_labels

    unique_clusters = sorted(
        [c for c in df_with_clusters["Cluster"].unique() if c != -1]
    )

    n_clusters = len(unique_clusters)
    cols = min(3, n_clusters)
    rows = (n_clusters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if n_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_clusters > 1 else axes

    fig.suptitle("Mapas Epiteliais Médios por Cluster", fontsize=18, fontweight="bold")

    for idx, cluster in enumerate(unique_clusters):
        cluster_data = df_with_clusters[df_with_clusters["Cluster"] == cluster]
        avg_data = {region: cluster_data[region].mean() for region in regions}

        ax = axes[idx] if n_clusters > 1 else axes[0]

        # Cria mapa
        plot_simple_map(
            ax,
            avg_data,
            {
                "pID": f"Cluster {cluster}",
                "Age": f"n={len(cluster_data)}",
                "Gender": f"μ={cluster_data[regions].mean().mean():.1f}",
                "Eye": f"σ={cluster_data[regions].std().mean():.1f}",
            },
        )

    # Remove eixos extras
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path / "cluster_average_maps.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Mapas médios dos {n_clusters} clusters salvos")


if __name__ == "__main__":
    # Carrega dados
    df = pd.read_csv("data/RTVue_20221110_MLClass.csv")

    print("\n" + "=" * 80)
    print("GERAÇÃO DE MAPAS EPITELIAIS")
    print("=" * 80)

    # Cria mapas de exemplo
    print("\n[1/2] Criando mapas de amostras individuais...")
    create_multiple_maps(df, n_samples=6)

    # Exemplo de mapa detalhado individual
    print("\n[2/2] Criando mapa detalhado de exemplo...")
    sample = df.iloc[0]
    regions = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
    data = {region: sample[region] for region in regions}

    title = (
        f"Mapeamento Epitelial - Paciente {sample.get('pID', 'N/A')}\n"
        f"Idade: {sample.get('Age', 'N/A')} anos | Sexo: {sample.get('Gender', 'N/A')} | Olho: {sample.get('Eye', 'N/A')}"
    )

    create_epithelial_map(
        data,
        title=title,
        save_path="results_epithelial_maps/example_detailed.png",
        cmap="RdYlGn_r",
    )

    print("\n✓ Todos os mapas foram gerados com sucesso!")
    print("✓ Resultados salvos em: results_epithelial_maps/")
