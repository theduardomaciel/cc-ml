# Atividade 01 - Pré-processamento

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./.github/cover.png">
  <source media="(prefers-color-scheme: light)" srcset="./.github/cover_light.png">
  <img alt="Atividade 01 - Pré-processamento" src="/.github/cover_light.png">
</picture>

<br />
<br />

Para tudo que for enviado, assumimos que você está de acordo com o código de honra abaixo:

## Código de Honra

> "Como membro da comunidade deste curso, não vou participar nem tolerar a desonestidade acadêmica."

<br />

## 🎯 Objetivo da Atividade

**Trabalhar o pré-processamento de dados para o algoritmo k-NN.**

<br />

## 📌 Descrição da Atividade

Nesta atividade, apresentamos duas versões de um mesmo programa em Python que lê um banco de dados com informações de mulheres descendentes do povo Pima — o qual, segundo a [Wikipédia](https://en.wikipedia.org/wiki/Pima_people), "\[...] são um povo nativo dos Estados Unidos da América que viviam às margens dos rios Gila e Sal, na parte sul do estado do Arizona."

As versões do programa são:

* `diabetes_csv.py`: trabalha com arquivos no formato `.csv`;
* `diabetes_xlsx.py`: trabalha com arquivos no formato `.xlsx` (Excel).

Você pode escolher **qual formato prefere utilizar** para realizar o pré-processamento dos dados.

A única modificação que precisa ser feita no código é a inserção da **chave individual da equipe**, substituindo o texto abaixo:

```python
DEV_KEY = "COLOCAR_SUA_KEY_AQUI"
```

Feito isso, o programa já estará completo em termos de estrutura, **mas ainda não funcional**, pois os dados precisam ser **pré-processados** para atender aos requisitos do algoritmo **k-NN**, garantindo um bom desempenho.

> A principal tarefa da equipe é **pré-processar os dados**, modificando os arquivos no formato escolhido, para deixá-los adequados ao funcionamento eficiente do algoritmo k-NN.

<br />

### 🧠 Como o programa funciona

Suponha que o formato escolhido seja `.csv`. O programa `diabetes_csv.py`:

1. Lê o arquivo `diabetes_dataset.csv` (base de treino);

2. Armazena os dados nos vetores:

   * `X`: características dos pacientes (ex: idade, glicose, pressão arterial etc.);
   * `y`: rótulo ou classe (se o paciente tem ou não diabetes).

3. Treina um modelo preditivo utilizando o algoritmo **k-NN**, com `k = 3`.

4. Usa esse modelo para classificar os dados contidos no arquivo `diabetes_app.csv` (base de teste).

5. As **previsões** são enviadas para o **servidor**, que:

   * Calcula a **acurácia** (porcentagem de acertos),
   * Retorna o resultado para o programa,
   * E armazena o **melhor desempenho obtido pela equipe** até o momento.

<br />

## 📚 Descrição da Base de Dados

Esse conjunto de dados, intitulado **"Pima Indians Diabetes Data Set"**, foi desenvolvido pelo [National Institute of Diabetes and Digestive and Kidney Diseases](https://www.niddk.nih.gov/). Seu objetivo é **prever se um paciente tem diabetes**, com base em dados clínicos.

Todos os pacientes do conjunto de dados são **mulheres com pelo menos 21 anos**, de **herança indígena Pima**.

### 🔎 Atributos do dataset:

1. **Pregnancies**: número de gestações
2. **Glucose**: concentração de glicose no plasma após 2h em teste oral de tolerância à glicose
3. **BloodPressure**: pressão arterial diastólica (mm Hg)
4. **SkinThickness**: espessura da dobra cutânea do tríceps (mm)
5. **Insulin**: insulina sérica de 2 horas (mu U/ml)
6. **BMI**: índice de massa corporal (peso em kg / altura em m²)
7. **DiabetesPedigreeFunction**: função de pedigree (hereditariedade de diabetes)
8. **Age**: idade (anos)
9. **Outcome**: resultado (0 = não diabético, 1 = diabético)

<br />

## 🐍 Preparando o Ambiente

### 1. Instale o Python

Se você ainda não possui o Python instalado:

#### 🔧 Windows

1. Acesse o site oficial: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Baixe a versão mais recente (recomendada: Python 3.10 ou superior)
3. Durante a instalação, **marque a opção "Add Python to PATH"**.
4. Após instalar, abra o terminal (cmd ou PowerShell) e execute:

```bash
python --version
```

Se tudo estiver correto, você verá a versão do Python instalada e estará pronto para rodar a atividade.


<details>
<summary>🐧 Linux</summary>

Use o gerenciador de pacotes da sua distribuição. Exemplo no Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-pip
```

</details>

<details>
<summary>🍏 macOS</summary>

Você pode instalar o Python usando o Homebrew. Se ainda não tiver o Homebrew instalado, siga as instruções em [https://brew.sh/](https://brew.sh/).

```bash
brew install python
```

</details>

---

### 2. Clone o repositório

Se ainda não fez isso, clone o repositório para sua máquina:

```bash
git clone https://github.com/theduardomaciel/cc-ml.git
cd cc-ml/01_Preprocessing/csv
```

### 3. Crie e ative um ambiente virtual (recomendado)


<details>
<summary>Windows (cmd ou PowerShell)</summary>

```bash
python -m venv venv
venv\Scripts\activate
```

</details>

<details>
<summary>Linux/macOS</summary>

```bash
python3 -m venv venv
source venv/bin/activate
```

</details>

### 4. Instale as dependências

Com o ambiente virtual ativado, instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

Pronto! Agora você pode rodar o script normalmente.

<br />

## 🔗 Links Úteis

* 📁 [Dataset no Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* 📄 [Artigo científico sobre o dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC8943493/)