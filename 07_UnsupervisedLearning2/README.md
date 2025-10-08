# Análise de Clustering - Espessura Epitelial# Análise de Clustering - Espessura Epitelial# Atividade 4 - Descoberta dos perfis/padrões de olhos



## 📝 Resumo



Análise comparativa de três algoritmos de clustering (K-means, DBSCAN e K-medoids) em dados de espessura epitelial da córnea.## 📝 Resumo<picture>



**Conclusão:** Os dados **NÃO possuem estrutura natural de clusters bem definidos**.  <source media="(prefers-color-scheme: dark)" srcset="./.github/cover.png">



## 📊 ResultadosAnálise comparativa de três algoritmos de clustering (K-means, DBSCAN e K-medoids) em dados de espessura epitelial da córnea.  <source media="(prefers-color-scheme: light)" srcset="./.github/cover_light.png">



| Algoritmo | Cluster Principal | Silhouette | Avaliação |  <img alt="Atividade 04 - Clusterização" src="/.github/cover_light.png">

|-----------|-------------------|------------|-----------|

| K-Means | 99.9% | 0.96 | ❌ Extremamente desequilibrado |**Conclusão:** Os dados **NÃO possuem estrutura natural de clusters bem definidos**.</picture>

| DBSCAN | 95.8% | 0.62 | ❌ Muito desequilibrado |

| K-Medoids | 40.0% | 0.13 | ⚠️ Equilibrado mas má separação |



## 🔍 Causa do Problema## 📊 Resultados## 📋 Sobre a atividade



1. **Outliers extremos** (25% dos dados com valores absurdos)

2. **Dados homogêneos** (50% entre 49-56 μm)

3. **Baixa correlação** entre features (0.07)| Algoritmo | Cluster Principal | Silhouette | Avaliação |Análise de **aprendizado não-supervisionado** para descobrir perfis/padrões de espessura epitelial em mapas oculares. O objetivo é identificar grupos naturais de olhos com características similares de espessura do epitélio usando o algoritmo **K-Means**.

4. **Variância distribuída** (8 PCs para 90%)

5. **População concentrada** no centro|-----------|-------------------|------------|-----------|



## 📂 Estrutura do Projeto| K-Means | 99.9% | 0.96 | ❌ Extremamente desequilibrado |### 🎯 Objetivo



```| DBSCAN | 95.8% | 0.62 | ❌ Muito desequilibrado |

07_UnsupervisedLearning2/

├── src/                           # Scripts Python| K-Medoids | 40.0% | 0.13 | ⚠️ Equilibrado mas má separação |Descobrir **quais são os perfis de espessura epitelial** que existem na base de dados, agrupando olhos com características similares.

│   ├── kmeans_clustering.py       # Implementação K-means

│   ├── dbscan_clustering.py       # Implementação DBSCAN

│   ├── kmedoids_clustering.py     # Implementação K-medoids

│   ├── diagnostic_analysis.py     # Análise diagnóstica## 🔍 Causa do Problema### 📊 Descrição dos Dados

│   ├── test_all_algorithms.py     # Teste comparativo

│   └── generate_comparison.py     # Visualização comparativa

├── docs/                          # Documentação

│   ├── ANALISE_PROBLEMA_CLUSTERING.md1. **Outliers extremos** (25% dos dados com valores absurdos)**Variáveis de Identificação:**

│   └── ESTRUTURA_SIMPLIFICADA.md

├── data/                          # Dataset2. **Dados homogêneos** (50% entre 49-56 μm)- `Index` = índice

│   └── RTVue_20221110_MLClass.csv

├── results/                       # Resultados e visualizações3. **Baixa correlação** entre features (0.07)- `pID` = ID do paciente

├── README.md                      # Este arquivo

└── requirements.txt               # Dependências4. **Variância distribuída** (8 PCs para 90%)- `Age` = idade

```

5. **População concentrada** no centro- `Gender` = gênero/sexo

## 🚀 Uso

- `Eye` = olho (OS=esquerdo; OD=direito)

```bash

# Análise diagnóstica## 📂 Arquivos

python src/diagnostic_analysis.py

**Variáveis de Espessura Epitelial (μm):**

# Testar todos os algoritmos

python src/test_all_algorithms.py- `kmeans_clustering.py` - Implementação K-means- `C` = central



# Gerar comparação visual- `dbscan_clustering.py` - Implementação DBSCAN- `S` = superior

python src/generate_comparison.py

```- `kmedoids_clustering.py` - Implementação K-medoids- `ST` = superior temporal



## 📋 Recomendações- `diagnostic_analysis.py` - Análise diagnóstica- `T` = temporal



1. Remover outliers antes de clustering- `test_all_algorithms.py` - Teste comparativo- `IT` = inferior temporal

2. Adicionar features demográficas (idade, gênero)

3. Considerar análise supervisionada se houver labels clínicos- `generate_comparison.py` - Visualização comparativa- `I` = inferior

4. Segmentar por características antes de aplicar clustering

- `ANALISE_PROBLEMA_CLUSTERING.md` - Análise detalhada- `IN` = inferior nasal

## 📖 Documentação Completa

- `N` = nasal

Para análise detalhada, ver `docs/ANALISE_PROBLEMA_CLUSTERING.md`

## 🚀 Uso- `SN` = superior nasal



```bash### 🧭 Distribuição Espacial (Rosa dos Ventos)

# Análise diagnóstica

python diagnostic_analysis.pyA disposição espacial das regiões no olho pode ser visualizada como:



# Testar todos os algoritmos|        |       |        |

python test_all_algorithms.py|--------|-------|--------|

| **ST** | **S** | **SN** |

# Gerar comparação visual| **T**  | **C** | **N**  |

python generate_comparison.py| **IT** | **I** | **IN** |

```

> 💡 Considere o enantiomorfismo dos olhos ao interpretar os resultados!

## 📋 Recomendações

## 🚀 Como Executar

1. Remover outliers antes de clustering

2. Adicionar features demográficas (idade, gênero)### 1️⃣ Instalação das Dependências

3. Considerar análise supervisionada se houver labels clínicos

4. Segmentar por características antes de aplicar clustering```bash

pip install -r requirements.txt

Para análise completa, ver `ANALISE_PROBLEMA_CLUSTERING.md````


### 2️⃣ Execução do Projeto

**✅ Opção Recomendada: Jupyter Notebook (Interativo e Visual)**

```bash
jupyter notebook analise_clustering.ipynb
```

O notebook contém:
- 📊 Exploração de dados interativa
- 🔍 Otimização visual do número de clusters
- 🎨 Visualizações profissionais
- 📈 Análise passo a passo
- 💾 Exportação de resultados

**Opção 2: Script Principal (CLI)**

```bash
python main.py
```

Oferece 5 modos de execução:
1. **Análise Completa** - Otimização + Clusterização + Apresentação
2. **Apenas Otimização** - Encontrar o melhor número de clusters
3. **Apenas Clusterização** - Executar K-Means com K específico
4. **Apenas Apresentação** - Gerar visualizações para o cliente
5. **Análise Rápida** - K=3 sem otimização (padrão)

**Opção 3: Importar Módulos (Para Scripts Customizados)**

```python
from kmeans_clustering import KMeansEpithelialClusterer
from optimization import KOptimizer
from presentation import ClientPresentation

# Seu código aqui...
```

## 📁 Estrutura do Projeto

```
07_UnsupervisedLearning2/
├── data/
│   ├── RTVue_20221110_MLClass.csv         # Dataset de espessura epitelial
│   └── RTVue_20221110_MLClass.xlsx        # Versão Excel
├── results/                                # Resultados gerados
│   ├── k_optimization.png                 # Análise de K ótimo
│   ├── kmeans_distributions.png           # Distribuições por cluster
│   ├── kmeans_profiles.png                # Perfis médios
│   ├── kmeans_results.csv                 # Resultados com labels
│   ├── k_optimization_metrics.csv         # Métricas de otimização
│   ├── 01_executive_summary.png           # Resumo executivo
│   ├── 02_detailed_profiles.png           # Perfis detalhados
│   └── 03_clinical_interpretation.png     # Interpretação clínica
├── 📓 analise_clustering.ipynb            # ⭐ Notebook principal (RECOMENDADO)
├── 🔧 kmeans_clustering.py                # Módulo: Classe de clusterização
├── 🔧 optimization.py                     # Módulo: Otimização de K
├── 🔧 presentation.py                     # Módulo: Apresentação ao cliente
├── 📜 main.py                             # Script CLI interativo
├── 📋 requirements.txt                    # Dependências
└── 📖 README.md                           # Documentação
```

### 🎯 Estrutura Modular

- **📓 Notebook**: Análise interativa com visualizações (uso recomendado)
- **🔧 Módulos Python**: Funções técnicas reutilizáveis
- **📜 Script CLI**: Execução automatizada via terminal

## 🔬 Metodologia

### 1. Pré-processamento
- Seleção das 9 variáveis de espessura epitelial
- Remoção de valores ausentes
- Normalização dos dados (StandardScaler)

### 2. Otimização do Número de Clusters
Utiliza 4 métricas para determinar o K ótimo:
- **Método do Cotovelo** (Elbow Method)
- **Silhouette Score** (maior é melhor)
- **Calinski-Harabasz Score** (maior é melhor)
- **Davies-Bouldin Score** (menor é melhor)

### 3. Clusterização
- Algoritmo: **K-Means**
- Inicialização: 10 execuções (n_init=10)
- Semente aleatória: 42 (reprodutibilidade)

### 4. Apresentação dos Resultados
Três visualizações principais:
1. **Resumo Executivo**: Distribuição, radar chart, comparação geral
2. **Perfis Detalhados**: Análise individual de cada cluster
3. **Interpretação Clínica**: Assimetrias, variabilidade, características

## 📊 Resultados Esperados

O projeto gera:

✅ **Perfis de Espessura**: Grupos de olhos com padrões similares  
✅ **Visualizações**: Gráficos profissionais para apresentação  
✅ **Métricas**: Avaliação quantitativa da qualidade dos clusters  
✅ **Dataset Rotulado**: CSV com cluster de cada olho  
✅ **Interpretação Clínica**: Insights sobre características distintivas

## 📈 Métricas de Avaliação

- **Silhouette Score**: Mede quão similar um ponto é ao seu cluster vs outros clusters
- **Calinski-Harabasz**: Razão da dispersão entre clusters e dentro dos clusters
- **Davies-Bouldin**: Média da similaridade entre cada cluster e seu mais similar
- **Inércia**: Soma das distâncias quadradas dos pontos aos centróides

## 💡 Dicas de Interpretação

1. **Perfis Espaciais**: Observe o padrão no radar chart (formato da "rosa")
2. **Espessura Central vs Periférica**: Compare região C com as demais
3. **Assimetrias**: Verifique diferenças Superior-Inferior e Temporal-Nasal
4. **Variabilidade**: Clusters com maior desvio padrão são mais heterogêneos

## 🔧 Personalização

Para ajustar o número de clusters, edite em `main.py`:
```python
clusterer = KMeansEpithelialClusterer(n_clusters=3)  # Altere o valor
```

## 📚 Dependências

- pandas >= 2.1.3
- numpy >= 1.26.2
- scikit-learn >= 1.3.2
- matplotlib >= 3.8.2
- seaborn >= 0.13.0

## 👥 Apresentação ao Cliente

Os arquivos gerados em `results/` são projetados para apresentação profissional:
- Gráficos de alta resolução (300 DPI)
- Paleta de cores consistente
- Interpretação clínica clara
- Métricas quantitativas e qualitativas