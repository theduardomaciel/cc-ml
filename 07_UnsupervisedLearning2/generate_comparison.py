"""
Gera um resumo visual comparativo dos três algoritmos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Configurar encoding para UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Configurações
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Dados dos resultados
results = {
    'Algoritmo': ['K-Means', 'DBSCAN', 'K-Medoids'],
    'Silhouette': [0.9620, 0.6229, 0.1278],
    'Calinski-Harabasz': [696.04, 315.53, 278.86],
    'Davies-Bouldin': [0.0574, 0.4720, 1.6165],
    'Cluster Principal (%)': [99.9, 95.8, 40.0],
    'Ruído/Outliers (%)': [0.1, 3.0, 0.0]
}

# Distribuições detalhadas
distributions = {
    'K-Means': [99.9, 0.0, 0.0],
    'DBSCAN': [95.8, 0.4, 0.4, 0.4],  # Excluindo ruído
    'K-Medoids': [40.0, 32.8, 27.2]
}

# Criar figura
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Título principal
fig.suptitle('Comparação de Algoritmos de Clustering\nK-means vs DBSCAN vs K-medoids', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Gráfico de barras - Distribuição do cluster principal
ax1 = fig.add_subplot(gs[0, :2])
algorithms = results['Algoritmo']
cluster_pcts = results['Cluster Principal (%)']
colors = ['#e74c3c', '#e67e22', '#27ae60']

bars = ax1.barh(algorithms, cluster_pcts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Percentual no Cluster Principal (%)', fontsize=12, fontweight='bold')
ax1.set_title('Problema: Concentração no Cluster Principal', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 105)

# Adicionar valores nas barras
for i, (bar, pct) in enumerate(zip(bars, cluster_pcts)):
    ax1.text(pct + 1, i, f'{pct:.1f}%', va='center', fontweight='bold', fontsize=11)

# Adicionar linha de referência
ax1.axvline(33.33, color='green', linestyle='--', linewidth=2, label='Ideal (33.3% para k=3)', alpha=0.7)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3, axis='x')

# 2. Métricas comparativas
ax2 = fig.add_subplot(gs[0, 2])
metrics_data = pd.DataFrame({
    'K-Means': [0.9620, 696.04, 0.0574],
    'DBSCAN': [0.6229, 315.53, 0.4720],
    'K-Medoids': [0.1278, 278.86, 1.6165]
}, index=['Silhouette\n(maior=melhor)', 'Calinski-Harabasz\n(maior=melhor)', 'Davies-Bouldin\n(menor=melhor)'])

# Normalizar para visualização
metrics_norm = metrics_data.copy()
metrics_norm.loc['Silhouette\n(maior=melhor)'] = metrics_data.loc['Silhouette\n(maior=melhor)']
metrics_norm.loc['Calinski-Harabasz\n(maior=melhor)'] = metrics_data.loc['Calinski-Harabasz\n(maior=melhor)'] / 700
metrics_norm.loc['Davies-Bouldin\n(menor=melhor)'] = 1 - (metrics_data.loc['Davies-Bouldin\n(menor=melhor)'] / 2)

sns.heatmap(metrics_norm, annot=metrics_data.values, fmt='.2f', cmap='RdYlGn', 
            center=0.5, ax=ax2, cbar_kws={'label': 'Performance Normalizada'}, 
            linewidths=1, linecolor='black')
ax2.set_title('Métricas de Qualidade', fontsize=12, fontweight='bold')
ax2.set_ylabel('')

# 3. Pizza charts - Distribuições
for idx, (name, dist) in enumerate(distributions.items()):
    ax = fig.add_subplot(gs[1, idx])
    
    # Preparar dados
    if name == 'DBSCAN':
        labels = ['Principal\n(95.8%)', 'C1 (0.4%)', 'C2 (0.4%)', 'C3 (0.4%)']
        colors_pie = ['#e74c3c', '#3498db', '#9b59b6', '#1abc9c']
    elif name == 'K-Means':
        labels = ['Principal\n(99.9%)', 'C1 (0.0%)', 'C2 (0.0%)']
        colors_pie = ['#e74c3c', '#3498db', '#9b59b6']
    else:
        labels = ['C0 (40.0%)', 'C1 (32.8%)', 'C2 (27.2%)']
        colors_pie = ['#27ae60', '#2ecc71', '#90ee90']
    
    # Criar pizza
    wedges, texts, autotexts = ax.pie(dist, labels=labels, autopct='', 
                                        colors=colors_pie, startangle=90,
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    ax.set_title(f'{name}', fontsize=13, fontweight='bold', pad=10)
    
    # Destacar problema ou sucesso
    if idx < 2:  # K-Means e DBSCAN
        ax.add_patch(plt.Circle((0, 0), 1.1, color='red', fill=False, linewidth=3, linestyle='--', alpha=0.5))
    else:  # K-Medoids
        ax.add_patch(plt.Circle((0, 0), 1.1, color='green', fill=False, linewidth=3, linestyle='--', alpha=0.5))

# 4. Análise das causas do problema
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

causas_texto = """
🔍 POR QUE TODOS OS ALGORITMOS FALHARAM EM ENCONTRAR PERFIS BEM DEFINIDOS?

1. 📊 OUTLIERS EXTREMOS (25% dos dados!)
   • Valores absurdos: C=770, S=7318, N=2310 μm
   • Forçam dados normais a se agruparem juntos

2. 🎯 DADOS MUITO HOMOGÊNEOS
   • 50% dos valores entre 49-56 μm (apenas 7 μm de diferença)
   • População uniforme sem subgrupos naturais

3. 🔗 BAIXA CORRELAÇÃO (média 0.07)
   • Features independentes, sem padrões coerentes
   • Não há "assinaturas" que definam grupos

4. 📉 VARIÂNCIA DISTRIBUÍDA
   • PCA: necessários 8 componentes para 90% da variância
   • Nenhuma direção privilegiada de separação

5. 📍 CONCENTRAÇÃO NO CENTRO
   • 75% dos pontos muito próximos do centroide
   • Sugere população homogênea

✅ MELHOR RESULTADO: K-MEDOIDS
   • Distribuição mais equilibrada (40%, 33%, 27%)
   • MAS: Silhouette Score = 0.13 (muito baixo!)
   • Diferenças entre medoides: apenas ~5-10 μm
   • Clusters NÃO são bem separados

❌ CONCLUSÃO: Os dados NÃO possuem estrutura natural de clusters distintos.
              Isso não é falha dos algoritmos, mas característica dos dados.

📋 RECOMENDAÇÕES:
   1. Remover outliers antes de clustering
   2. Investigar valores extremos (erros de medição?)
   3. Adicionar features demográficas (idade, gênero, etc.)
   4. Considerar análise supervisionada se houver labels clínicos
   5. Segmentar por características antes de clustering
"""

ax4.text(0.5, 0.5, causas_texto, fontsize=10, verticalalignment='center',
         horizontalalignment='center', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Salvar
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
plt.savefig(results_dir / "comparative_analysis.png", dpi=300, bbox_inches='tight')
print("✅ Visualização comparativa salva em: results/comparative_analysis.png")
plt.close()

# Criar tabela de comparação em texto
print("\n" + "=" * 80)
print("RESUMO COMPARATIVO")
print("=" * 80)

df_results = pd.DataFrame(results)
print("\n", df_results.to_string(index=False))

print("\n" + "=" * 80)
print("VENCEDOR: K-MEDOIDS (distribuição mais equilibrada)")
print("PROBLEMA: Silhouette Score muito baixo (0.13) indica má separação")
print("=" * 80)
