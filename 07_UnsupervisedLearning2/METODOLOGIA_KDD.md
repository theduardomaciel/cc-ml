# 📚 Metodologia KDD Aplicada

## O que é KDD?

**KDD (Knowledge Discovery in Databases)** é um processo iterativo e interativo de descoberta de conhecimento em bases de dados. É a metodologia completa que envolve várias etapas para extrair padrões úteis e compreensíveis dos dados.

## Diferença entre KDD e Data Mining

- **KDD**: Processo completo (da seleção dos dados à interpretação)
- **Data Mining**: Uma das etapas do KDD (aplicação de algoritmos)

## 🔄 Etapas do KDD Aplicadas Neste Projeto

### 1️⃣ Seleção de Dados (Data Selection)

**Objetivo**: Identificar e selecionar os dados relevantes para a análise.

**Neste projeto**:
- Dataset: `RTVue_20221110_MLClass.csv`
- Variáveis selecionadas: 9 medidas de espessura epitelial
  - `C` (Central), `S` (Superior), `ST` (Superior Temporal)
  - `T` (Temporal), `IT` (Inferior Temporal), `I` (Inferior)
  - `IN` (Inferior Nasal), `N` (Nasal), `SN` (Superior Nasal)
- Variáveis descartadas: Index, pID, Age, Gender, Eye (não relevantes para descobrir perfis)

**Implementação**:
```python
features = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
df_selected = df[features]
```

---

### 2️⃣ Pré-processamento (Data Preprocessing)

**Objetivo**: Limpar e preparar os dados para análise.

**Neste projeto**:
- **Tratamento de valores ausentes**: Remoção de registros incompletos
- **Verificação de outliers**: Análise de distribuições
- **Validação de dados**: Garantir valores positivos e razoáveis

**Problemas encontrados e soluções**:
- Valores ausentes (NaN) → Removidos
- ~0.5% dos dados removidos (mantendo 99.5% do dataset)

**Implementação**:
```python
df_clean = df_selected.dropna()  # Remove valores ausentes
```

---

### 3️⃣ Transformação (Data Transformation)

**Objetivo**: Transformar os dados em formato adequado para o algoritmo de mineração.

**Neste projeto**:
- **Normalização**: StandardScaler (z-score normalization)
  - Transforma cada variável para média 0 e desvio padrão 1
  - Essencial para K-Means (algoritmo baseado em distância)
  
**Por que normalizar?**
- K-Means usa distância euclidiana
- Variáveis com diferentes escalas podem dominar o cálculo
- Todas as regiões têm importância similar (40-70 μm)

**Implementação**:
```python
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df_clean)
```

**Resultado**:
- Média: 0.000000
- Desvio padrão: 1.000000
- Todas as variáveis na mesma escala

---

### 4️⃣ Mineração de Dados (Data Mining)

**Objetivo**: Aplicar algoritmos de aprendizado de máquina para descobrir padrões.

**Neste projeto**:
- **Algoritmo**: K-Means Clustering
- **Por que K-Means?**
  - Adequado para dados numéricos
  - Eficiente para grandes volumes
  - Resultados interpretáveis (centróides = perfis médios)
  - Não-supervisionado (descoberta de padrões)

**Otimização do K**:
Antes de aplicar K-Means, otimizamos o número de clusters usando 4 métricas:

1. **Método do Cotovelo (Elbow Method)**
   - Analisa a inércia vs número de clusters
   - Procura o "cotovelo" na curva

2. **Silhouette Score** (0.45 - 0.75 = bom)
   - Mede quão similar um ponto é ao seu cluster vs outros
   - Varia de -1 a 1 (quanto maior, melhor)

3. **Calinski-Harabasz Score**
   - Razão da dispersão entre clusters / dentro dos clusters
   - Quanto maior, melhor

4. **Davies-Bouldin Score**
   - Média da similaridade entre cada cluster e seu mais similar
   - Quanto menor, melhor

**Implementação**:
```python
# Otimização
optimizer = KOptimizer(k_range=(2, 11))
df_metrics, k_optimal = optimizer.optimize(df_clean)

# Clusterização
kmeans = KMeans(n_clusters=k_optimal, n_init=10, random_state=42)
labels = kmeans.fit_predict(data_normalized)
```

**Parâmetros do K-Means**:
- `n_clusters`: número de clusters (otimizado)
- `n_init=10`: 10 execuções com diferentes inicializações
- `random_state=42`: reprodutibilidade

---

### 5️⃣ Interpretação/Avaliação (Interpretation/Evaluation)

**Objetivo**: Interpretar os padrões descobertos e avaliar sua qualidade.

**Neste projeto**:

#### Análise Quantitativa:
- Tamanho de cada cluster (número de olhos)
- Perfis médios de espessura (centróides)
- Desvios padrão (variabilidade)
- Assimetrias (Superior-Inferior, Temporal-Nasal)

#### Análise Qualitativa:
- Interpretação clínica dos perfis
- Identificação de características distintivas
- Nomeação dos clusters (ex: "fino", "normal", "espesso")

#### Visualizações Geradas:
1. **Radar Charts**: Perfis espaciais dos clusters
2. **Boxplots**: Distribuições por região
3. **Heatmaps**: Correlações e mapas espaciais
4. **Gráficos de assimetria**: Superior-Inferior, Temporal-Nasal
5. **Interpretação clínica**: Texto explicativo

**Exemplo de interpretação**:
```
Cluster 0 (45% dos olhos):
- Espessura NORMAL (média: 54.2 μm)
- Padrão UNIFORME (C vs P: 0.8 μm)
- Simetria S-I PRESERVADA (+1.2 μm)
- HOMOGÊNEO (DP: 3.5 μm)
```

---

## 🔄 Iteratividade do KDD

O KDD não é linear! Podemos retornar a etapas anteriores:

```
┌─────────────┐
│  Seleção    │
└──────┬──────┘
       ↓
┌──────────────┐
│ Pré-proc.    │ ← Voltar se houver problemas
└──────┬───────┘
       ↓
┌──────────────┐
│ Transformação│ ← Testar outras normalizações
└──────┬───────┘
       ↓
┌──────────────┐
│ Mineração    │ ← Testar outros K ou algoritmos
└──────┬───────┘
       ↓
┌──────────────┐
│ Interpretação│ ← Avaliar e refinar
└──────────────┘
```

**Exemplos de iteração neste projeto**:
- Se K=3 não for satisfatório → Testar K=4 ou K=5
- Se normalização não funcionar → Testar Min-Max ou Robust Scaler
- Se clusters não fizerem sentido → Revisar seleção de variáveis

---

## 📊 Métricas de Avaliação Usadas

### 1. Silhouette Score
$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

- `a(i)`: distância média intra-cluster
- `b(i)`: distância média ao cluster mais próximo
- Interpretação:
  - `s > 0.7`: Estrutura forte
  - `0.5 < s < 0.7`: Estrutura razoável
  - `0.25 < s < 0.5`: Estrutura fraca
  - `s < 0.25`: Sem estrutura

### 2. Calinski-Harabasz Score
$$CH = \frac{SS_B / (k-1)}{SS_W / (n-k)}$$

- `SS_B`: soma dos quadrados entre clusters
- `SS_W`: soma dos quadrados dentro dos clusters
- Maior valor = melhor separação

### 3. Davies-Bouldin Score
$$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{i \neq j} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$

- `σ_i`: dispersão média do cluster i
- `d(c_i, c_j)`: distância entre centróides
- Menor valor = melhor separação

### 4. Inércia (Within-Cluster Sum of Squares)
$$I = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

- Soma das distâncias quadradas aos centróides
- Menor valor = clusters mais compactos
- Usado no método do cotovelo

---

## 🎯 Diferenciais da Implementação

1. **Modularidade**: Código organizado em classes reutilizáveis
2. **Seguimento rigoroso do KDD**: Cada etapa claramente separada
3. **Múltiplas métricas**: Decisão baseada em consenso
4. **Visualizações profissionais**: Adequadas para apresentação clínica
5. **Interpretação clínica**: Tradução dos resultados técnicos para linguagem médica
6. **Reprodutibilidade**: `random_state=42` em todos os lugares
7. **Documentação**: Código comentado e mensagens informativas

---

## 📚 Referências

- Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). *From data mining to knowledge discovery in databases*. AI magazine, 17(3), 37-37.

- Han, J., Kamber, M., & Pei, J. (2011). *Data mining: concepts and techniques*. Elsevier.

- Kaufman, L., & Rousseeuw, P. J. (2009). *Finding groups in data: an introduction to cluster analysis*. John Wiley & Sons.

- Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*. Journal of computational and applied mathematics, 20, 53-65.

---

**Elaborado para a disciplina de Machine Learning - UFAL**
