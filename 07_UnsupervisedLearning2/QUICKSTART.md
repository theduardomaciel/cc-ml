# 🚀 Guia Rápido - Atividade 4

## ⚡ Início Rápido

### Opção 1: Notebook (Recomendado) 📓
```bash
jupyter notebook analise_clustering.ipynb
```
✅ **Melhor para**: Exploração interativa, visualizações, apresentação

### Opção 2: Script CLI 💻
```bash
python main.py
```
✅ **Melhor para**: Execução automatizada, batch processing

---

## 📚 Estrutura dos Arquivos

### 🎯 Para Usar:
- **`analise_clustering.ipynb`** → Análise completa passo a passo
- **`main.py`** → Execução via terminal

### 🔧 Módulos Técnicos (não executar diretamente):
- **`kmeans_clustering.py`** → Classe `KMeansEpithelialClusterer`
- **`optimization.py`** → Classe `KOptimizer`
- **`presentation.py`** → Classe `ClientPresentation`

---

## 🎨 Fluxo de Trabalho

1. **Explorar dados** → No notebook
2. **Otimizar K** → Métricas + gráficos
3. **Clusterizar** → K-Means com K ótimo
4. **Visualizar** → Apresentações profissionais
5. **Exportar** → CSV + imagens

---

## 💡 Dicas

- 🔍 Use o notebook para entender o processo
- 📊 As visualizações são salvas automaticamente em `results/`
- 🎯 Ajuste `K_OPTIMAL` no notebook conforme necessário
- 💾 Resultados em CSV incluem todos os dados + cluster labels

---

## ❓ Problemas Comuns

**Erro ao importar módulos?**
```bash
# Certifique-se de estar no diretório correto
cd 07_UnsupervisedLearning2
```

**Faltando dependências?**
```bash
pip install -r requirements.txt
```

**Notebook não abre?**
```bash
pip install jupyter notebook
```
