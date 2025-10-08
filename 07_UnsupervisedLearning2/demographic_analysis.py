"""
Análise Demográfica e Clínica
Explora os campos Age, Gender e Eye em conjunto com os clusters de espessura epitelial
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from preprocessing import load_and_preprocess

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300


def analyze_demographics(file_path, n_clusters=3, output_dir="results_demographic"):
    """
    Análise demográfica completa dos clusters

    Args:
        file_path: Caminho do arquivo CSV
        n_clusters: Número de clusters para K-Means
        output_dir: Diretório para salvar resultados
    """
    print("\n" + "=" * 80)
    print("ANÁLISE DEMOGRÁFICA E CLÍNICA")
    print("=" * 80)

    # Carrega e preprocessa
    print("\n[1/5] Carregando dados...")
    df_clean, scaled_data, features, scaler = load_and_preprocess(
        file_path, remove_outliers=True
    )

    # Clustering para análise
    print("\n[2/5] Executando clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clean["Cluster"] = kmeans.fit_predict(scaled_data)

    # Cria diretório de saída
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Análises
    print("\n[3/5] Análise por Idade...")
    age_analysis = analyze_age_distribution(df_clean, output_path)

    print("\n[4/5] Análise por Gênero...")
    gender_analysis = analyze_gender_distribution(df_clean, output_path)

    print("\n[5/5] Análise de Lateralidade (Olhos)...")
    eye_analysis = analyze_eye_laterality(df_clean, features, output_path)

    # Visualização integrada
    create_integrated_visualization(df_clean, features, output_path)

    # Relatório estatístico
    generate_statistical_report(
        df_clean, features, age_analysis, gender_analysis, eye_analysis, output_path
    )

    print(f"\n✓ Análise completa salva em: {output_path.absolute()}")
    return df_clean, age_analysis, gender_analysis, eye_analysis


def analyze_age_distribution(df, output_path):
    """Analisa distribuição etária por cluster"""
    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Análise de Distribuição Etária por Cluster", fontsize=16, fontweight="bold"
    )

    # 1. Boxplot de idade por cluster
    ax = axes[0, 0]
    df.boxplot(column="Age", by="Cluster", ax=ax)
    ax.set_title("Distribuição de Idade por Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Idade (anos)")
    plt.sca(ax)
    plt.xticks(rotation=0)

    # 2. Histograma sobreposto
    ax = axes[0, 1]
    for cluster in sorted(df["Cluster"].unique()):
        cluster_data = df[df["Cluster"] == cluster]["Age"]
        ax.hist(
            cluster_data,
            bins=20,
            alpha=0.5,
            label=f"Cluster {cluster}",
            edgecolor="black",
        )
    ax.set_xlabel("Idade (anos)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de Idade Sobreposta")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Violin plot
    ax = axes[1, 0]
    sns.violinplot(data=df, x="Cluster", y="Age", ax=ax, palette="Set2")
    ax.set_title("Densidade de Idade por Cluster (Violin Plot)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Idade (anos)")

    # 4. Tabela estatística
    ax = axes[1, 1]
    ax.axis("off")

    stats_data = []
    for cluster in sorted(df["Cluster"].unique()):
        cluster_ages = df[df["Cluster"] == cluster]["Age"]
        stats_data.append(
            [
                f"Cluster {cluster}",
                f"{cluster_ages.mean():.1f}",
                f"{cluster_ages.median():.1f}",
                f"{cluster_ages.std():.1f}",
                f"{cluster_ages.min():.0f}-{cluster_ages.max():.0f}",
                f"{len(cluster_ages)}",
            ]
        )

    table = ax.table(
        cellText=stats_data,
        colLabels=["Cluster", "Média", "Mediana", "Desvio", "Range", "N"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Colorir header
    for i in range(6):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.tight_layout()
    plt.savefig(output_path / "age_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Testes estatísticos
    print("\n  Estatísticas por Cluster:")
    clusters = sorted(df["Cluster"].unique())
    for cluster in clusters:
        cluster_ages = df[df["Cluster"] == cluster]["Age"]
        print(
            f"    Cluster {cluster}: {cluster_ages.mean():.1f} ± {cluster_ages.std():.1f} anos (n={len(cluster_ages)})"
        )

    # ANOVA
    if len(clusters) > 2:
        groups = [df[df["Cluster"] == c]["Age"].values for c in clusters]
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"\n  ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
        results["anova"] = {"f_stat": f_stat, "p_value": p_value}

        if p_value < 0.05:
            print("  ⚠ Diferença significativa de idade entre clusters!")

    results["stats"] = stats_data
    return results


def analyze_gender_distribution(df, output_path):
    """Analisa distribuição de gênero por cluster"""
    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Análise de Distribuição de Gênero por Cluster", fontsize=16, fontweight="bold"
    )

    # 1. Contagem por cluster e gênero
    ax = axes[0, 0]
    gender_cluster = pd.crosstab(df["Cluster"], df["Gender"])
    gender_cluster.plot(kind="bar", ax=ax, color=["#FF6B9D", "#4ECDC4"], alpha=0.8)
    ax.set_title("Distribuição de Gênero por Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Contagem")
    ax.legend(title="Gênero", labels=["Feminino", "Masculino"])
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # 2. Proporções (stacked bar)
    ax = axes[0, 1]
    gender_pct = pd.crosstab(df["Cluster"], df["Gender"], normalize="index") * 100
    gender_pct.plot(
        kind="bar", stacked=True, ax=ax, color=["#FF6B9D", "#4ECDC4"], alpha=0.8
    )
    ax.set_title("Proporção de Gênero por Cluster (%)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percentual (%)")
    ax.legend(title="Gênero", labels=["Feminino", "Masculino"])
    ax.set_ylim(0, 100)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # 3. Heatmap de contagem
    ax = axes[1, 0]
    sns.heatmap(
        gender_cluster,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Contagem"},
    )
    ax.set_title("Heatmap: Cluster vs Gênero")
    ax.set_xlabel("Gênero")
    ax.set_ylabel("Cluster")

    # 4. Tabela com chi-square test
    ax = axes[1, 1]
    ax.axis("off")

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(gender_cluster)

    table_data = []
    for cluster in sorted(df["Cluster"].unique()):
        f_count = gender_cluster.loc[cluster, "F"]
        m_count = gender_cluster.loc[cluster, "M"]
        total = f_count + m_count
        f_pct = 100 * f_count / total
        m_pct = 100 * m_count / total

        table_data.append(
            [
                f"Cluster {cluster}",
                f"{f_count} ({f_pct:.1f}%)",
                f"{m_count} ({m_pct:.1f}%)",
                f"{total}",
            ]
        )

    # Adiciona totais
    total_f = gender_cluster["F"].sum()
    total_m = gender_cluster["M"].sum()
    total_all = total_f + total_m
    table_data.append(
        [
            "TOTAL",
            f"{total_f} ({100*total_f/total_all:.1f}%)",
            f"{total_m} ({100*total_m/total_all:.1f}%)",
            f"{total_all}",
        ]
    )

    table = ax.table(
        cellText=table_data,
        colLabels=["Cluster", "Feminino", "Masculino", "Total"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 0.7],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Colorir header
    for i in range(4):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Adiciona resultado do teste
    test_text = f"Chi-Square Test\nχ² = {chi2:.4f}, p = {p_value:.4f}\n"
    if p_value < 0.05:
        test_text += "Associação significativa entre\ncluster e gênero (p < 0.05)"
    else:
        test_text += "Sem associação significativa\nentre cluster e gênero"

    ax.text(
        0.5,
        0.15,
        test_text,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path / "gender_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n  Chi-Square Test: χ²={chi2:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("  ⚠ Associação significativa entre cluster e gênero!")

    results["chi2"] = {"chi2": chi2, "p_value": p_value, "dof": dof}
    results["distribution"] = gender_cluster
    return results


def analyze_eye_laterality(df, features, output_path):
    """Analisa lateralidade (OD vs OS) e assimetrias"""
    results = {}

    # Calcula médias por olho
    df["Mean_Thickness"] = df[features].mean(axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Análise de Lateralidade (Olho Direito vs Esquerdo)",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Distribuição de olhos por cluster
    ax = axes[0, 0]
    eye_cluster = pd.crosstab(df["Cluster"], df["Eye"])
    eye_cluster.plot(kind="bar", ax=ax, color=["#95E1D3", "#F38181"], alpha=0.8)
    ax.set_title("Distribuição de Lateralidade por Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Contagem")
    ax.legend(title="Olho", labels=["Esquerdo (OS)", "Direito (OD)"])
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # 2. Comparação de espessura média OD vs OS
    ax = axes[0, 1]
    eye_groups = df.groupby("Eye")["Mean_Thickness"].apply(list)
    bp = ax.boxplot(
        [eye_groups["OD"], eye_groups["OS"]],
        labels=["OD (Direito)", "OS (Esquerdo)"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["#F38181", "#95E1D3"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Espessura Média (μm)")
    ax.set_title("Comparação de Espessura OD vs OS")
    ax.grid(True, alpha=0.3, axis="y")

    # T-test
    od_thickness = df[df["Eye"] == "OD"]["Mean_Thickness"]
    os_thickness = df[df["Eye"] == "OS"]["Mean_Thickness"]
    t_stat, p_value = stats.ttest_ind(od_thickness, os_thickness)

    # 3. Distribuição por cluster e olho
    ax = axes[0, 2]
    for cluster in sorted(df["Cluster"].unique()):
        for eye, color in [("OD", "#F38181"), ("OS", "#95E1D3")]:
            data = df[(df["Cluster"] == cluster) & (df["Eye"] == eye)]["Mean_Thickness"]
            if len(data) > 0:
                ax.scatter(
                    [cluster] * len(data),
                    data,
                    alpha=0.3,
                    s=20,
                    color=color,
                    label=f"{eye} (C{cluster})" if cluster == 0 else "",
                )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Espessura Média (μm)")
    ax.set_title("Espessura por Cluster e Lateralidade")
    ax.grid(True, alpha=0.3)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#F38181",
            markersize=8,
            label="OD",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#95E1D3",
            markersize=8,
            label="OS",
        ),
    ]
    ax.legend(handles=handles)

    # 4. Heatmap de features por lateralidade
    ax = axes[1, 0]
    eye_means = df.groupby("Eye")[features].mean()
    sns.heatmap(
        eye_means.T,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        ax=ax,
        cbar_kws={"label": "Espessura (μm)"},
    )
    ax.set_title("Espessura Média por Região e Lateralidade")
    ax.set_ylabel("Região Epitelial")
    ax.set_xlabel("Olho")

    # 5. Diferença absoluta entre OD e OS por região
    ax = axes[1, 1]
    eye_diff = (eye_means.loc["OD"] - eye_means.loc["OS"]).abs()
    colors = ["#e74c3c" if x > 1 else "#3498db" for x in eye_diff]
    bars = ax.bar(
        range(len(features)), eye_diff, color=colors, alpha=0.7, edgecolor="black"
    )
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45)
    ax.set_ylabel("|Diferença| (μm)")
    ax.set_title("Assimetria OD-OS por Região")
    ax.axhline(
        y=1, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Limiar 1μm"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Adiciona valores nas barras
    for bar, val in zip(bars, eye_diff):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 6. Tabela estatística
    ax = axes[1, 2]
    ax.axis("off")

    table_data = [
        ["Métrica", "OD (Direito)", "OS (Esquerdo)"],
        ["Amostras", f"{len(od_thickness)}", f"{len(os_thickness)}"],
        [
            "Espessura Média",
            f"{od_thickness.mean():.2f} μm",
            f"{os_thickness.mean():.2f} μm",
        ],
        [
            "Desvio Padrão",
            f"{od_thickness.std():.2f} μm",
            f"{os_thickness.std():.2f} μm",
        ],
        [
            "Mín - Máx",
            f"{od_thickness.min():.1f} - {od_thickness.max():.1f}",
            f"{os_thickness.min():.1f} - {os_thickness.max():.1f}",
        ],
    ]

    table = ax.table(
        cellText=table_data, cellLoc="center", loc="center", bbox=[0, 0.3, 1, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Colorir header
    for i in range(3):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Resultado do t-test
    test_text = f"T-Test (OD vs OS)\nt = {t_stat:.4f}, p = {p_value:.4f}\n"
    if p_value < 0.05:
        test_text += "Diferença significativa\nentre OD e OS (p < 0.05)"
    else:
        test_text += "Sem diferença significativa\nentre OD e OS"

    ax.text(
        0.5,
        0.1,
        test_text,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path / "eye_laterality.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n  T-Test (OD vs OS): t={t_stat:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("  ⚠ Diferença significativa de espessura entre OD e OS!")

    print(f"\n  Assimetrias > 1μm:")
    for feature, diff in zip(features, eye_diff):
        if diff > 1:
            print(f"    {feature}: {diff:.2f} μm")

    results["t_test"] = {"t_stat": t_stat, "p_value": p_value}
    results["asymmetry"] = dict(zip(features, eye_diff))
    results["eye_means"] = eye_means
    return results


def create_integrated_visualization(df, features, output_path):
    """Cria visualização integrada de todas as variáveis demográficas"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle(
        "Análise Demográfica Integrada por Cluster", fontsize=18, fontweight="bold"
    )

    clusters = sorted(df["Cluster"].unique())
    colors = plt.cm.Set3(range(len(clusters)))

    # Para cada cluster, criar painel
    for idx, cluster in enumerate(clusters):
        cluster_data = df[df["Cluster"] == cluster]

        row = idx
        base_color = colors[idx]

        # 1. Distribuição de idade
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.hist(
            cluster_data["Age"], bins=15, color=base_color, alpha=0.7, edgecolor="black"
        )
        ax1.set_title(f"Cluster {cluster}: Idade")
        ax1.set_xlabel("Idade (anos)")
        ax1.set_ylabel("Frequência")
        ax1.axvline(
            cluster_data["Age"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'Média: {cluster_data["Age"].mean():.1f}',
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Proporção de gênero
        ax2 = fig.add_subplot(gs[row, 1])
        gender_counts = cluster_data["Gender"].value_counts()
        wedges, texts, autotexts = ax2.pie(
            gender_counts.values,
            labels=[
                "Feminino" if g == "F" else "Masculino" for g in gender_counts.index
            ],
            autopct="%1.1f%%",
            colors=["#FF6B9D", "#4ECDC4"],
            startangle=90,
        )
        ax2.set_title(f"Cluster {cluster}: Gênero")
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        # 3. Proporção de olhos
        ax3 = fig.add_subplot(gs[row, 2])
        eye_counts = cluster_data["Eye"].value_counts()
        wedges, texts, autotexts = ax3.pie(
            eye_counts.values,
            labels=[
                "Direito (OD)" if e == "OD" else "Esquerdo (OS)"
                for e in eye_counts.index
            ],
            autopct="%1.1f%%",
            colors=["#F38181", "#95E1D3"],
            startangle=90,
        )
        ax3.set_title(f"Cluster {cluster}: Lateralidade")
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        # 4. Perfil médio de espessura
        ax4 = fig.add_subplot(gs[row, 3])
        mean_profile = cluster_data[features].mean()
        std_profile = cluster_data[features].std()

        x_pos = range(len(features))
        ax4.bar(
            x_pos,
            mean_profile,
            yerr=std_profile,
            color=base_color,
            alpha=0.7,
            edgecolor="black",
            capsize=5,
        )
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(features, rotation=45, ha="right")
        ax4.set_ylabel("Espessura (μm)")
        ax4.set_title(f"Cluster {cluster}: Perfil Epitelial")
        ax4.grid(True, alpha=0.3, axis="y")

    plt.savefig(output_path / "integrated_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Visualização integrada criada")


def generate_statistical_report(
    df, features, age_analysis, gender_analysis, eye_analysis, output_path
):
    """Gera relatório estatístico em texto"""

    report = []
    report.append("=" * 80)
    report.append("RELATÓRIO ESTATÍSTICO - ANÁLISE DEMOGRÁFICA")
    report.append("=" * 80)
    report.append("")

    # Resumo geral
    report.append("RESUMO GERAL")
    report.append("-" * 80)
    report.append(f"Total de amostras: {len(df)}")
    report.append(f"Número de clusters: {df['Cluster'].nunique()}")
    report.append(f"Idade média: {df['Age'].mean():.1f} ± {df['Age'].std():.1f} anos")
    report.append(f"Range de idade: {df['Age'].min():.0f} - {df['Age'].max():.0f} anos")
    report.append(
        f"Gênero: {(df['Gender']=='F').sum()} F ({100*(df['Gender']=='F').sum()/len(df):.1f}%), "
        f"{(df['Gender']=='M').sum()} M ({100*(df['Gender']=='M').sum()/len(df):.1f}%)"
    )
    report.append(
        f"Lateralidade: {(df['Eye']=='OD').sum()} OD, {(df['Eye']=='OS').sum()} OS"
    )
    report.append("")

    # Análise por cluster
    report.append("ANÁLISE POR CLUSTER")
    report.append("-" * 80)
    for cluster in sorted(df["Cluster"].unique()):
        cluster_data = df[df["Cluster"] == cluster]
        report.append(
            f"\nCluster {cluster} (n={len(cluster_data)}, {100*len(cluster_data)/len(df):.1f}%):"
        )
        report.append(
            f"  Idade: {cluster_data['Age'].mean():.1f} ± {cluster_data['Age'].std():.1f} anos"
        )
        report.append(
            f"  Gênero: {(cluster_data['Gender']=='F').sum()} F, {(cluster_data['Gender']=='M').sum()} M"
        )
        report.append(
            f"  Lateralidade: {(cluster_data['Eye']=='OD').sum()} OD, {(cluster_data['Eye']=='OS').sum()} OS"
        )

        mean_thickness = cluster_data[features].mean().mean()
        std_thickness = cluster_data[features].std().mean()
        report.append(
            f"  Espessura média: {mean_thickness:.2f} ± {std_thickness:.2f} μm"
        )

    report.append("")

    # Testes estatísticos
    report.append("TESTES ESTATÍSTICOS")
    report.append("-" * 80)

    if "anova" in age_analysis:
        anova = age_analysis["anova"]
        report.append(f"\n1. ANOVA - Idade entre clusters:")
        report.append(f"   F-statistic: {anova['f_stat']:.4f}")
        report.append(f"   p-value: {anova['p_value']:.4f}")
        report.append(
            f"   Resultado: {'Diferença significativa' if anova['p_value'] < 0.05 else 'Sem diferença significativa'}"
        )

    chi2_result = gender_analysis["chi2"]
    report.append(f"\n2. Chi-Square - Gênero vs Cluster:")
    report.append(f"   χ²: {chi2_result['chi2']:.4f}")
    report.append(f"   p-value: {chi2_result['p_value']:.4f}")
    report.append(f"   Graus de liberdade: {chi2_result['dof']}")
    report.append(
        f"   Resultado: {'Associação significativa' if chi2_result['p_value'] < 0.05 else 'Sem associação significativa'}"
    )

    t_test = eye_analysis["t_test"]
    report.append(f"\n3. T-Test - OD vs OS:")
    report.append(f"   t-statistic: {t_test['t_stat']:.4f}")
    report.append(f"   p-value: {t_test['p_value']:.4f}")
    report.append(
        f"   Resultado: {'Diferença significativa' if t_test['p_value'] < 0.05 else 'Sem diferença significativa'}"
    )

    report.append("")

    # Assimetrias
    report.append("ASSIMETRIAS ENTRE OLHOS (|OD - OS|)")
    report.append("-" * 80)
    asymmetries = eye_analysis["asymmetry"]
    for feature in sorted(
        asymmetries.keys(), key=lambda x: asymmetries[x], reverse=True
    ):
        diff = asymmetries[feature]
        flag = " ⚠" if diff > 1 else ""
        report.append(f"  {feature:3s}: {diff:.3f} μm{flag}")

    report.append("")
    report.append("=" * 80)

    # Salva relatório
    report_text = "\n".join(report)
    with open(output_path / "statistical_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n✓ Relatório estatístico gerado")
    print("\n" + report_text)


if __name__ == "__main__":
    data_path = "data/RTVue_20221110_MLClass.csv"

    df_analyzed, age_results, gender_results, eye_results = analyze_demographics(
        data_path, n_clusters=3, output_dir="results_demographic"
    )

    print("\n" + "=" * 80)
    print("ANÁLISE COMPLETA!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("  - age_distribution.png")
    print("  - gender_distribution.png")
    print("  - eye_laterality.png")
    print("  - integrated_analysis.png")
    print("  - statistical_report.txt")
