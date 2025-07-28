# Atividade 01 - Pré-processamento

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./.github/cover.png">
  <source media="(prefers-color-scheme: light)" srcset="./.github/cover_light.png">
  <img alt="Atividade 01 - Pré-processamento" src="/.github/cover_light.png">
</picture>

<br />
<br />

Para tudo que nos é enviado, assumimos que você está seguindo o código de honra a seguir.

## Código de Honra

>"Como membro da comunidade deste curso, não vou participar nem tolerar a desonestidade acadêmica".

## Objetivo da atividade
*Trabalhar o pré-processamento de dados para o algoritmo k-NN*

## Descrição da atividade
Nesta atividade apresentamos duas versões de um mesmo programa em Python que lê um banco de dados de mulheres descendentes do povo Pima, o qual, segundo a [Wikipédia](https://en.wikipedia.org/wiki/Pima_people), "[...] são um povo nativo dos Estados Unidos da América que viviam às margens dos rios Gila e Sal, na parte sul do estado de Arizona."

A primeira versão do programa Python (`diabetes_csv.py`) trabalha com arquivos de dados em formato `.csv` e a segunda com os arquivos em formato Excel (`diabetes_xlsx.py`). Deste modo você pode escolher o formato que preferir para trabalhar com os dados e fazer seu pré-processamento.

A única modificação que precisa ser realizada no programa Python é a inserção da chave individual da equipe, substituindo o texto entre aspas.

```python
DEV_KEY = "COLOCAR_SUA_KEY_AQUI"
```

Feito isso o programa já está completo, porém ainda não funcional, pois existem erros nos dados que precisam ser pré-processados para que cumpram tudo que é demandado pelo algoritmo k-NN para que este funcione bem.

> A atividade da equipe consiste em **pré-processar os dados**, modificando os arquivos no formato escolhido, para que estes se encontrem da melhor maneira para o funcionamento do algoritmo k-NN.

Supondo que o formato escolhido tenha sido o .csv, o programa `diabetes_csv.py` lê o arquivo `diabetes_dataset.csv` (base de treino) e o armazena nos vetores `X` e `y`, em que:
- X: dados das características (ex: idade, nível de glicose, pressão);
- y: rótulo de saída (se tem ou não diabetes).

Em seguida ele constrói o modelo preditivo utilizando o k-NN com um `k = 3` e utiliza esse modelo para classificar os dados encontrados no arquivo `diabetes_app.csv` (base de teste).

Tais previsões são enviadas para o servidor que vai contabilizar a acurácia (porcentagem de acertos) conseguida com as previsões realizadas, para em seguida retornar e armazenar o melhor desempenho conseguido pela equipe.

## Descrição da base de dados

Esse conjunto de dados, intitulado "Pima Indians Diabetes Data Set", foi desenvolvido pelo [National Institute of Diabetes and Digestive and Kidney Diseases](https://www.niddk.nih.gov/), cujo objetivo é prever se o paciente tem diabetes. 

Os pacientes selecionados são mulheres com pelo menos 21 anos e de herança indiana Prima. As informações da base de dados são descritas a seguir.

#### Atributos do dataset:
1. **Pregnancies**: número de gestações
2. **Glucose**: concentração plasmática de glicose a 2 horas em um teste oral de tolerância à glicose (Concentração de glicose no plasma)
3. **BloodPressure**: pressão sanguínea diastólica (mm Hg)
4. **SkinThickness**: espessura da dobra cutânea do tríceps (mm)
5. **Insulin**: insulina sérica de 2 horas (mu U/ml)
6. **BMI**: índice de massa corporal (peso em kg / (altura em m) ^ 2)
7. **DiabetesPedigreeFunction**: função de pedigree do diabetes (hereditariedade)
8. **Age**: idade (anos)
9. **Outcome**: resultado, ou seja, variável de classe (0 ou 1) para diabetes

## Instalando o Python



## Links Úteis

-   [Dataset no Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
-   [Artigo interessante sobre o Dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC8943493/)