# Atividade 4 - Descoberta dos perfis/padrões de olhos

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./.github/cover.png">
  <source media="(prefers-color-scheme: light)" srcset="./.github/cover_light.png">
  <img alt="Atividade 04 - Clusterização" src="/.github/cover_light.png">
</picture>

## 📋 Sobre a atividade

Análise de **aprendizado não-supervisionado** para descobrir perfis/padrões de espessura epitelial em mapas oculares. O objetivo é identificar grupos naturais de olhos com características similares de espessura do epitélio usando o algoritmo **K-Means**.

### 🎯 Objetivo

Descobrir **quais são os perfis de espessura epitelial** que existem na base de dados, agrupando olhos com características similares.

### 📊 Descrição dos Dados

**Variáveis de Identificação:**
- `Index` = índice
- `pID` = ID do paciente
- `Age` = idade
- `Gender` = gênero/sexo
- `Eye` = olho (OS=esquerdo; OD=direito)

**Variáveis de Espessura Epitelial (μm):**
- `C` = central
- `S` = superior
- `ST` = superior temporal
- `T` = temporal
- `IT` = inferior temporal
- `I` = inferior
- `IN` = inferior nasal
- `N` = nasal
- `SN` = superior nasal

### 🧭 Distribuição Espacial (Rosa dos Ventos)

A disposição espacial das regiões no olho pode ser visualizada como:

|        |       |        |
|--------|-------|--------|
| **ST** | **S** | **SN** |
| **T**  | **C** | **N**  |
| **IT** | **I** | **IN** |

> 💡 Considere o enantiomorfismo dos olhos ao interpretar os resultados!

## 🚀 Como Executar

### 1️⃣ Instalação das Dependências

```bash
pip install -r requirements.txt
```

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