"""
Script de Análise Diagnóstica dos Dados de Espessura Epitelial
Objetivo: Entender por que os algoritmos de clustering não estão encontrando perfis diprint("=" * 80)
print("8. GERANDO VISUALIZAÇÕES DIAGNÓSTICAS")
print("=" * 80)

RESULTS_DIR.mkdir(exist_ok=True)

# 8.1. Distribuições
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import io

# Configurar encoding para UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Configurações
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

# Determina o diretório base (pai de src/)
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "RTVue_20221110_MLClass.csv"
RESULTS_DIR = BASE_DIR / "results"

# Carrega os dados
print("=" * 80)
print("ANÁLISE DIAGNÓSTICA - DADOS DE ESPESSURA EPITELIAL")
print("=" * 80)

data = pd.read_csv(DATA_PATH)
features = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]
df_features = data[features].dropna()

print(f"\n📊 Informações Gerais:")
print(f"  - Total de amostras: {len(df_features)}")
print(f"  - Número de features: {len(features)}")

# 1. ESTATÍSTICAS DESCRITIVAS
print("\n" + "=" * 80)
print("1. ESTATÍSTICAS DESCRITIVAS DAS FEATURES")
print("=" * 80)
print(df_features.describe())

# 2. ANÁLISE DE VARIABILIDADE
print("\n" + "=" * 80)
print("2. ANÁLISE DE VARIABILIDADE")
print("=" * 80)

cv = df_features.std() / df_features.mean() * 100  # Coeficiente de variação
print(f"\nCoeficiente de Variação (%):")
for feat in features:
    print(f"  {feat:>3}: {cv[feat]:>6.2f}%")
print(f"\n  Média do CV: {cv.mean():.2f}%")

# 3. ANÁLISE DE OUTLIERS
print("\n" + "=" * 80)
print("3. ANÁLISE DE OUTLIERS (usando IQR)")
print("=" * 80)

outliers_count = {}
for col in features:
    Q1 = df_features[col].quantile(0.25)
    Q3 = df_features[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_features[
        (df_features[col] < lower_bound) | (df_features[col] > upper_bound)
    ][col]
    outliers_count[col] = len(outliers)
    if len(outliers) > 0:
        print(
            f"\n{col}: {len(outliers)} outliers ({len(outliers)/len(df_features)*100:.2f}%)"
        )
        print(f"  Valores: min={outliers.min()}, max={outliers.max()}")
        print(f"  Range normal: [{lower_bound:.2f}, {upper_bound:.2f}]")

total_outliers = sum(outliers_count.values())
print(f"\nTotal de outliers detectados: {total_outliers}")

# 4. ANÁLISE DE CORRELAÇÃO
print("\n" + "=" * 80)
print("4. ANÁLISE DE CORRELAÇÃO ENTRE FEATURES")
print("=" * 80)

corr_matrix = df_features.corr()
print("\nMédia das correlações (excluindo diagonal):")
mean_corr = (corr_matrix.sum().sum() - len(features)) / (
    len(features) * (len(features) - 1)
)
print(f"  {mean_corr:.4f}")

print("\nMaiores correlações:")
corr_pairs = []
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        corr_pairs.append((features[i], features[j], corr_matrix.iloc[i, j]))
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for feat1, feat2, corr in corr_pairs[:5]:
    print(f"  {feat1} <-> {feat2}: {corr:.4f}")

# 5. ANÁLISE DE DISTRIBUIÇÃO
print("\n" + "=" * 80)
print("5. ANÁLISE DE DISTRIBUIÇÃO (Teste de Normalidade)")
print("=" * 80)

for col in features:
    stat, p_value = stats.shapiro(df_features[col])
    is_normal = "✓ Normal" if p_value > 0.05 else "✗ Não-normal"
    print(f"  {col:>3}: p-value = {p_value:.6f} {is_normal}")

# 6. ANÁLISE DE HOMOGENEIDADE
print("\n" + "=" * 80)
print("6. ANÁLISE DE HOMOGENEIDADE DOS DADOS")
print("=" * 80)

# Normaliza os dados
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

# Calcula a distância de cada ponto ao centroide
centroid = scaled_data.mean(axis=0)
distances = np.sqrt(((scaled_data - centroid) ** 2).sum(axis=1))

print(f"\nDistâncias ao centroide:")
print(f"  Média: {distances.mean():.4f}")
print(f"  Desvio padrão: {distances.std():.4f}")
print(f"  Mínima: {distances.min():.4f}")
print(f"  Máxima: {distances.max():.4f}")

# Percentis de distância
percentiles = [25, 50, 75, 90, 95, 99]
print(f"\nPercentis de distância:")
for p in percentiles:
    print(f"  {p}º: {np.percentile(distances, p):.4f}")

# 7. ANÁLISE PCA
print("\n" + "=" * 80)
print("7. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
print("=" * 80)

pca = PCA()
pca.fit(scaled_data)

print(f"\nVariância explicada por componente:")
cumulative_var = 0
for i, var in enumerate(pca.explained_variance_ratio_):
    cumulative_var += var
    print(f"  PC{i+1}: {var*100:.2f}% (acumulado: {cumulative_var*100:.2f}%)")

# 8. VISUALIZAÇÕES
print("\n" + "=" * 80)
print("8. GERANDO VISUALIZAÇÕES DIAGNÓSTICAS")
print("=" * 80)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 8.1. Distribuições
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle(
    "Distribuições das Features de Espessura Epitelial", fontsize=16, fontweight="bold"
)

for idx, col in enumerate(features):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df_features[col], bins=50, edgecolor="black", alpha=0.7)
    ax.set_title(
        f"{col} (μ={df_features[col].mean():.1f}, σ={df_features[col].std():.1f})"
    )
    ax.set_xlabel("Espessura (μm)")
    ax.set_ylabel("Frequência")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "diagnostic_distributions.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ Distribuições salvas")

# 8.2. Boxplots
fig, ax = plt.subplots(figsize=(14, 6))
df_features.boxplot(ax=ax)
ax.set_title(
    "Boxplots das Features de Espessura Epitelial", fontsize=14, fontweight="bold"
)
ax.set_ylabel("Espessura (μm)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "diagnostic_boxplots.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ Boxplots salvos")

# 8.3. Mapa de correlação
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Mapa de Correlação entre Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "diagnostic_correlation.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ Mapa de correlação salvo")

# 8.4. PCA
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Scree plot
ax1.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_ * 100,
    "bo-",
)
ax1.set_xlabel("Componente Principal")
ax1.set_ylabel("Variância Explicada (%)")
ax1.set_title("Scree Plot - Variância Explicada por Componente")
ax1.grid(True, alpha=0.3)

# Scatter plot dos primeiros 2 componentes
pca_2d = PCA(n_components=2).fit_transform(scaled_data)
ax2.scatter(pca_2d[:, 0], pca_2d[:, 1], alpha=0.5, s=10)
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax2.set_title("Projeção nos 2 Primeiros Componentes Principais")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "diagnostic_pca.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ Análise PCA salva")

# 8.5. Distâncias ao centroide
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(distances, bins=50, edgecolor="black", alpha=0.7)
ax.axvline(
    distances.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Média: {distances.mean():.2f}",
)
ax.axvline(
    np.median(distances),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Mediana: {np.median(distances):.2f}",
)
ax.set_xlabel("Distância ao Centroide")
ax.set_ylabel("Frequência")
ax.set_title("Distribuição das Distâncias ao Centroide (dados normalizados)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "diagnostic_distances.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ Análise de distâncias salva")

print("\n" + "=" * 80)
print("Análise completa! Visualizações salvas em 'results/'")
print("=" * 80)
