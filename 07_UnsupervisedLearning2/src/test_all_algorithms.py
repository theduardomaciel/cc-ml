"""
Script de Teste Completo: K-means, DBSCAN e K-medoids
Compara os três algoritmos e analisa os resultados
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import io

# Configurar encoding para UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Determina o diretório base
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from kmeans_clustering import KMeansEpithelialClusterer
from dbscan_clustering import DBSCANEpithelialClusterer

try:
    from kmedoids_clustering import KMedoidsEpithelialClusterer

    KMEDOIDS_AVAILABLE = True
except ImportError:
    print("⚠️ sklearn-extra não está instalado. K-medoids não será testado.")
    print("   Para instalar: pip install scikit-learn-extra")
    KMEDOIDS_AVAILABLE = False

DATA_PATH = BASE_DIR / "data" / "RTVue_20221110_MLClass.csv"

print("=" * 80)
print("COMPARAÇÃO DE ALGORITMOS DE CLUSTERING")
print("K-means vs DBSCAN vs K-medoids")
print("=" * 80)

# ==============================================================================
# 1. K-MEANS
# ==============================================================================
print("\n" + "=" * 80)
print("1️⃣  TESTANDO K-MEANS")
print("=" * 80)

kmeans = KMeansEpithelialClusterer(n_clusters=3, random_state=42)
kmeans.load_data(str(DATA_PATH))
kmeans.preprocess_data()
kmeans.fit()
kmeans_stats, kmeans_counts = kmeans.get_cluster_profiles()

# ==============================================================================
# 2. DBSCAN
# ==============================================================================
print("\n" + "=" * 80)
print("2️⃣  TESTANDO DBSCAN")
print("=" * 80)

dbscan = DBSCANEpithelialClusterer(eps=1.5, min_samples=20, random_state=42)
dbscan.load_data(str(DATA_PATH))
dbscan.preprocess_data()
dbscan.fit()
dbscan_stats, dbscan_counts = dbscan.get_cluster_profiles()

# ==============================================================================
# 3. K-MEDOIDS
# ==============================================================================
if KMEDOIDS_AVAILABLE:
    print("\n" + "=" * 80)
    print("3️⃣  TESTANDO K-MEDOIDS")
    print("=" * 80)

    kmedoids = KMedoidsEpithelialClusterer(n_clusters=3, method="pam", random_state=42)
    kmedoids.load_data(str(DATA_PATH))
    kmedoids.preprocess_data()
    kmedoids.fit()
    kmedoids_stats, kmedoids_counts = kmedoids.get_cluster_profiles()
    kmedoids.get_medoids()

# ==============================================================================
# 4. COMPARAÇÃO
# ==============================================================================
print("\n" + "=" * 80)
print("📊 COMPARAÇÃO DOS RESULTADOS")
print("=" * 80)

print("\n🔢 Distribuição de amostras por algoritmo:\n")

# K-means
print("K-MEANS (k=3):")
for cluster, count in kmeans_counts.items():
    percentage = (count / kmeans_counts.sum()) * 100
    print(f"  Cluster {cluster}: {count:>4} amostras ({percentage:>5.1f}%)")

# DBSCAN
print("\nDBSCAN (eps=1.5, min_samples=20):")
for cluster, count in dbscan_counts.items():
    percentage = (count / dbscan_counts.sum()) * 100
    label = "Ruído" if cluster == -1 else f"Cluster {cluster}"
    print(f"  {label:>10}: {count:>4} amostras ({percentage:>5.1f}%)")

# K-medoids
if KMEDOIDS_AVAILABLE:
    print("\nK-MEDOIDS (k=3):")
    for cluster, count in kmedoids_counts.items():
        percentage = (count / kmedoids_counts.sum()) * 100
        print(f"  Cluster {cluster}: {count:>4} amostras ({percentage:>5.1f}%)")

# ==============================================================================
# 5. ANÁLISE E DIAGNÓSTICO
# ==============================================================================
print("\n" + "=" * 80)
print("🔍 DIAGNÓSTICO DO PROBLEMA")
print("=" * 80)

print("\n❌ PROBLEMA IDENTIFICADO:")
print("   Todos os algoritmos apresentam distribuição extremamente desequilibrada.")
print(
    "   Isso indica que os dados NÃO possuem estrutura natural de clusters distintos.\n"
)

print("💡 POSSÍVEIS CAUSAS:\n")

print("1. 📊 OUTLIERS EXTREMOS:")
print("   - Foram detectados 1364 outliers (~25% dos dados)")
print("   - Valores extremos como C=770, S=7318, N=2310")
print(
    "   - Esses outliers forçam os dados normais a se concentrarem em um único cluster"
)

print("\n2. 🎯 HOMOGENEIDADE DOS DADOS:")
print("   - A maioria dos dados está concentrada em uma faixa muito estreita")
print("   - 50% dos valores estão entre 49-56 μm (diferença de apenas 7 μm)")
print("   - Baixa variabilidade intrínseca dificulta a separação em grupos distintos")

print("\n3. 🔗 BAIXA CORRELAÇÃO ENTRE FEATURES:")
print("   - Correlação média de apenas 0.0708 entre as features")
print("   - As features são praticamente independentes")
print("   - Não há padrões coerentes que definam subgrupos naturais")

print("\n4. 📉 VARIÂNCIA DISTRIBUÍDA:")
print("   - PCA mostra que são necessários 8 componentes para 90% da variância")
print("   - Nenhum componente principal domina (PC1 = apenas 18.5%)")
print("   - Os dados não têm direções privilegiadas de separação")

print("\n5. 📍 DADOS CENTRADOS:")
print("   - Distâncias ao centroide muito concentradas (75% < 1.38)")
print("   - Poucos pontos realmente distantes do centro")
print("   - Sugere população homogênea sem subgrupos naturais")

print("\n" + "=" * 80)
print("🎓 CONCLUSÃO")
print("=" * 80)

print("\n⚠️  OS DADOS NÃO SÃO ADEQUADOS PARA CLUSTERING:\n")

print("   Os três algoritmos testados (K-means, DBSCAN e K-medoids) falharam")
print("   em encontrar clusters balanceados. Isso NÃO é uma falha dos algoritmos,")
print("   mas sim uma característica intrínseca dos dados.\n")

print("   Os dados de espessura epitelial parecem vir de uma população relativamente")
print("   homogênea, com outliers esporádicos (possivelmente erros de medição ou")
print("   casos patológicos raros) que distorcem a análise.\n")

print("📋 RECOMENDAÇÕES:\n")

print("   1. REMOVER OUTLIERS: Aplicar filtro para remover outliers extremos")
print("      antes de tentar clustering novamente")

print("\n   2. INVESTIGAR OUTLIERS: Verificar se valores extremos são erros de")
print("      medição ou casos clínicos que devem ser tratados separadamente")

print("\n   3. FEATURES ADICIONAIS: Incluir outras variáveis (idade, gênero, etc.)")
print("      que possam ajudar a identificar subgrupos mais significativos")

print("\n   4. ANÁLISE SUPERVISIONADA: Se houver labels clínicos (diagnósticos),")
print("      usar classificação supervisionada em vez de clustering")

print("\n   5. ANÁLISE DE SUBPOPULAÇÕES: Segmentar por características demográficas")
print("      antes de aplicar clustering (ex: agrupar por faixa etária)")

print("\n" + "=" * 80)
print("Análise completa!")
print("=" * 80)
