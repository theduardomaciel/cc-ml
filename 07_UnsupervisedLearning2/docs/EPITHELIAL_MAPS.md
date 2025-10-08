# Mapas Epiteliais - Visualização de Espessura Corneana

## 📊 O que são os Mapas Epiteliais?

Os **mapas epiteliais** são visualizações circulares que representam a espessura do epitélio corneano em diferentes regiões do olho, similar aos mapas topográficos usados em exames oftalmológicos (como o RTVue OCT).

## 🎯 Anatomia do Mapa

O mapa é dividido em **9 regiões anatômicas**:

### Região Central
- **C** (Centro): região central da córnea

### Regiões Periféricas (sentido horário a partir do topo)
- **S** (Superior): região superior
- **ST** (Superotemporal): superior-temporal
- **T** (Temporal): lateral externa
- **IT** (Inferotemporal): inferior-temporal  
- **I** (Inferior): região inferior
- **IN** (Inferonasal): inferior-nasal
- **N** (Nasal): lado do nariz
- **SN** (Superonasal): superior-nasal

```
       S
    SN   ST
  N    C    T
    IN   IT
       I
```

## 🌈 Esquema de Cores


Os mapas utilizam um **gradiente de cores** invertido (RdYlGn_r - Red-Yellow-Green invertido):

- 🟢 **Verde**: Epitélio mais FINO (valores baixos)
- 🟡 **Amarelo**: Espessura MÉDIA
- 🔴 **Vermelho**: Epitélio mais ESPESSO (valores altos)

A barra de cores (colorbar) indica a escala de espessura em **micrômetros (μm)**.

## 📁 Arquivos Gerados

### 1. Mapas de Amostras Individuais
`sample_epithelial_maps.png`
- 6 exemplos aleatórios de pacientes
- Útil para ver a variabilidade individual

`detailed_map_1.png`, `detailed_map_2.png`, `detailed_map_3.png`
- Mapas detalhados com informações completas do paciente
- Mostra: ID do paciente, idade, sexo, olho (OD/OS)

### 2. Mapas Médios por Cluster
Gerados para cada algoritmo (KMeans, DBSCAN, K-Medoids):

`cluster_average_maps.png`
- Mapa médio de cada cluster encontrado
- Revela padrões de espessura característicos de cada grupo
- Informações mostradas:
  - **n**: número de amostras no cluster
  - **μ**: espessura média geral
  - **σ**: desvio padrão médio

## 🔍 Como Interpretar

### Padrões Normais vs. Anormais


**Padrão Normal:**
- Distribuição relativamente uniforme
- Verde predominante (epitélio mais fino/saudável)
- Pequenas variações entre regiões

**Padrões Patológicos:**
- Assimetrias pronunciadas
- Regiões vermelhas (espessamento)
- Grandes diferenças entre regiões adjacentes

### Uso no Clustering

Os mapas médios por cluster ajudam a:

1. **Identificar subgrupos clínicos**
   - Clusters podem representar diferentes condições oculares
   - Ex: olhos normais vs. com patologias

2. **Validar resultados do clustering**
   - Clusters distintos devem ter mapas visivelmente diferentes
   - Clusters similares podem indicar over-fitting

3. **Descobrir padrões anatômicos**
   - Regiões específicas mais espessas/finas em certos grupos
   - Assimetrias características de condições específicas

## 💡 Aplicações Clínicas

### Diagnóstico
- Detecção de ceratocone (afinamento progressivo)
- Identificação de ectasias corneanas
- Monitoramento pós-cirúrgico

### Pesquisa
- Comparação entre populações
- Estudos de progressão de doenças
- Validação de tratamentos

## 🔬 Exemplo de Análise


**Cluster com epitélio mais fino:**
```
Cluster 0: μ=52.3 μm
- Regiões I, IT mais verdes
- Possível indicação de afinamento inferior
- Pode sugerir ceratocone subclínico
```

**Cluster com epitélio mais espesso:**
```
Cluster 1: μ=58.7 μm  
- Vermelho predominante em todas regiões
- Distribuição uniforme
- Provavelmente olhos saudáveis
```

## 📊 Código de Uso

### Gerar mapas individuais
```python
from epithelial_mapping import create_epithelial_map

data = {
    'C': 55.0, 'S': 56.0, 'ST': 60.0,
    'T': 71.0, 'IT': 57.0, 'I': 54.0,
    'IN': 52.0, 'N': 52.0, 'SN': 51.0
}

create_epithelial_map(
    data, 
    title="Paciente #1234",
    save_path="mapa.png"
)
```

### Gerar mapas por cluster
```python
from epithelial_mapping import create_cluster_average_maps

create_cluster_average_maps(
    df=dataframe_com_medidas,
    cluster_labels=labels_do_kmeans,
    output_dir="resultados/"
)
```

## 📚 Referências

- RTVue OCT (Optovue) - Tecnologia de Tomografia de Coerência Óptica
- Anatomia da córnea e regiões de análise padrão
- Valores de referência para espessura epitelial normal: ~50-60 μm

---

**Desenvolvido para análise de dados de espessura epitelial corneana**  
Atividade de Machine Learning - UFAL 2025