# Análise de Clustering - Espessura Epitelial 👁️

## 📝 Descrição

Este projeto analisa dados de espessura epitelial da córnea usando algoritmos de clustering não supervisionado. O objetivo inicial era identificar perfis distintos de espessura epitelial, mas a análise revelou que **os dados não possuem estrutura natural de clusters bem definidos**.

## 🔬 Algoritmos Testados

1. **K-Means** - Algoritmo clássico baseado em centróides
2. **DBSCAN** - Algoritmo baseado em densidade, robusto a outliers
3. **K-Medoids** - Variação do K-Means usando medoides, mais robusto a outliers

## 📊 Resultados Principais

| Algoritmo | Cluster Principal | Silhouette Score | Distribuição |
|-----------|-------------------|------------------|--------------|
| **K-Means** | 99.9% | 0.9620 (enganoso) | ❌ Extremamente desequilibrada |
| **DBSCAN** | 95.8% | 0.6229 | ❌ Muito desequilibrada |
| **K-Medoids** | 40.0% | 0.1278 | ⚠️ Equilibrada mas má separação |

### Distribuições Detalhadas

**K-Means (k=3):**
- Cluster 0: 5284 amostras (99.9%)
- Cluster 1: 2 amostras (0.0%)
- Cluster 2: 1 amostra (0.0%)

**DBSCAN (eps=1.5, min_samples=20):**
- Cluster principal: 5067 amostras (95.8%)
- Ruído: 157 amostras (3.0%)
- Clusters menores: 20-23 amostras cada

**K-Medoids (k=3):**
- Cluster 0: 2117 amostras (40.0%)
- Cluster 1: 1732 amostras (32.8%)
- Cluster 2: 1438 amostras (27.2%)
- **Medoides:** ~49μm, ~53μm, ~58μm (diferenças mínimas)

## 🔍 Diagnóstico do Problema

### Por que todos os algoritmos falharam?

1. **📊 Outliers Extremos (25% dos dados)**
   - Valores absurdos: C=770, S=7318, N=2310 μm
   - Forçam dados normais a se concentrarem em um único cluster

2. **🎯 Homogeneidade dos Dados**
   - 50% dos valores entre 49-56 μm (apenas 7 μm de diferença)
   - População uniforme sem subgrupos naturais

3. **🔗 Baixa Correlação entre Features (média: 0.07)**
   - Features praticamente independentes
   - Sem padrões coerentes que definam grupos

4. **📉 Variância Distribuída Uniformemente**
   - PCA: necessários 8 componentes para 90% da variância
   - Nenhuma direção privilegiada de separação

5. **📍 Dados Concentrados no Centro**
   - 75% dos pontos muito próximos do centroide
   - Sugere população homogênea

## ✅ Melhor Resultado: K-Medoids

**Por que K-Medoids teve melhor distribuição?**
- Usa medoides (pontos reais) em vez de centróides (médias)
- Mais robusto a outliers
- Minimiza distâncias absolutas, não quadráticas

**Por que ainda assim falhou?**
- Silhouette Score = 0.1278 (muito baixo! <0.5 indica má separação)
- Davies-Bouldin = 1.6165 (alto = clusters sobrepostos)
- Diferenças entre medoides: apenas ~5-10 μm (clinicamente insignificante)

## 🎓 Conclusão

**❌ Os dados NÃO são adequados para clustering não supervisionado**

Os três algoritmos confirmam que os dados vêm de uma **população homogênea** sem subgrupos naturais baseados apenas nas medidas de espessura epitelial.

## 📋 Recomendações

### 1. 🧹 Pré-processamento
```python
# Remover outliers usando IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data_clean = data[(data >= Q1 - 1.5*IQR) & (data <= Q3 + 1.5*IQR)]
```

### 2. 🔬 Investigação de Outliers
- Verificar se são erros de medição
- Analisar casos patológicos separadamente

### 3. ➕ Features Adicionais
- Incluir idade, gênero, histórico clínico
- Outras medidas oculares

### 4. 🎯 Análise Supervisionada
- Se houver diagnósticos clínicos, usar classificação
- Random Forest, SVM, etc.

### 5. 📊 Segmentação Prévia
```python
# Agrupar por faixa etária antes de clustering
jovens = data[data['Age'] < 30]
adultos = data[(data['Age'] >= 30) & (data['Age'] < 60)]
idosos = data[data['Age'] >= 60]
```

## 📂 Estrutura do Projeto

```
07_UnsupervisedLearning2/
│
├── data/
│   └── RTVue_20221110_MLClass.csv       # Dataset original
│
├── results/
│   ├── kmeans_results.csv               # Resultados K-means
│   ├── dbscan_results.csv               # Resultados DBSCAN
│   ├── diagnostic_*.png                 # Visualizações diagnósticas
│   └── comparative_analysis.png         # Comparação visual dos algoritmos
│
├── kmeans_clustering.py                 # Implementação K-means
├── dbscan_clustering.py                 # Implementação DBSCAN
├── kmedoids_clustering.py               # Implementação K-medoids
├── diagnostic_analysis.py               # Análise diagnóstica completa
├── test_all_algorithms.py               # Teste comparativo
├── generate_comparison.py               # Gera visualização comparativa
│
├── ANALISE_PROBLEMA_CLUSTERING.md       # Análise detalhada do problema
└── README.md                            # Este arquivo
```

## 🚀 Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
pip install scikit-learn-extra  # Para K-medoids
```

### 2. Executar análise diagnóstica
```bash
python diagnostic_analysis.py
```

### 3. Testar todos os algoritmos
```bash
python test_all_algorithms.py
```

### 4. Gerar visualização comparativa
```bash
python generate_comparison.py
```

## 📊 Visualizações Geradas

1. **diagnostic_distributions.png** - Histogramas de todas as features
2. **diagnostic_boxplots.png** - Boxplots mostrando outliers
3. **diagnostic_correlation.png** - Mapa de correlação entre features
4. **diagnostic_pca.png** - Análise de componentes principais
5. **diagnostic_distances.png** - Distribuição de distâncias ao centroide
6. **comparative_analysis.png** - Comparação visual dos três algoritmos

## 📖 Documentação Adicional

- **ANALISE_PROBLEMA_CLUSTERING.md** - Análise detalhada de por que o clustering falhou
- Inclui explicações técnicas, métricas e recomendações práticas

## 🔧 Tecnologias Utilizadas

- Python 3.13
- pandas - Manipulação de dados
- numpy - Operações numéricas
- scikit-learn - Algoritmos K-means e DBSCAN
- scikit-learn-extra - Algoritmo K-medoids
- matplotlib/seaborn - Visualizações
- scipy - Testes estatísticos

## 👨‍💻 Autor

Eduardo Maciel

## 📅 Data

Outubro de 2025
