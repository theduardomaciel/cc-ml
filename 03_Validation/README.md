# Atividade 03 - Avaliação de classificadores

Para tudo que nos é enviado, assumimos que você está seguindo o código de honra a seguir.

## Código de Honra

>"Como membro da comunidade deste curso, não vou participar nem tolerar a desonestidade acadêmica".

## Objetivo da atividade
*Trabalhar a metodologia e as técnicas para a avaliação de classificadores*

## Descrição da atividade
A atividade da equipe consiste em construir e validar um modelo(s) preditivo(s) com o intuito de garantir o seu poder de generalização, utilizando a metodologia e técnicas vistas em sala de aula.

Nessa atividade vocês poderão utilizar qualquer algoritmo de aprendizagem mesmo os ainda não vistos em sala de aula tais como: SVM, Redes Neurais, RBFs, etc. Um detalhe importante é que o entendimento do algoritmo é necessário pois as equipes melhores ranqueadas terão, como nas outras atividades, que compartilhar o que foi aprendido e isso inclui o algoritmo utilizado.

Para o envio da atividade poderão ser utilizados os mesmos modelos de programas para ler e enviar os resultados utilizados na Atividade 01 - Pré-processamento, fazendo as devidas modificações como por exemplo alterar a URL de envio para https://aydanomachado.com/mlclass/03_Validation.php.

**Atenção:** nessa atividade só será permitido **1 envio a cada 12h** pois o objetivo é fazer uma boa validação do modelo antes desse ser enviado.

Ainda nos mesmos moldes da Atividade 01 os arquivos `abalone_dataset.xlsx` ou `abalone_dataset.csv` devem ser utilizados para a construção e validação do modelo preditivo (classificador) e os arquivos `abalone_app.xlsx` ou `abalone_app.csv` utilizados para teste do modelo.
As previsões deverão ser enviadas para o servidor onde será registrado o desempenho do modelo construído, correspondente a sua acurácia.

## Descrição da base de dados

Esse conjunto de dados foi modificado a partir da base encontrada no [UCI Machine Learning Repository: Abalone Data Set](http://archive.ics.uci.edu/ml/datasets/Abalone).
Que foi originalmente utilizado no estudo Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and Wes B Ford (1994) "The Population Biology of Abalone (Haliotis species) in Tasmania. I. Blacklip Abalone (H. rubra) from the North Coast and Islands of Bass Strait", Sea Fisheries Division, Technical Report No. 48 (ISSN 1034-3288).

A base consiste de informações de um molusco chamado Abalone, e o objetivo do classificador é identificar o tipo do exemplar (entre as classes I, II e III) utilizando as informações fornecidas e detalhadas a seguir.

#### Abalone
(Origem: [Wikipédia, a enciclopédia livre](https://pt.wikipedia.org/wiki/Abalone))
"Haliotis (popularmente conhecidos em português e inglês por abalone, também em inglês por ear shell ou ormer, em espanhol por oreja de mar e abulone, em francês por oreille de mer, em italiano por abaloni e em alemão por seeohren) é um gênero de moluscos gastrópodes marinhos da família Haliotidae e o único gênero catalogado desta família. Foi proposto por Linnaeus em 1758 e contém diversas espécies em águas costeiras de quase todo o mundo. Na gastronomia, o abalone é um molusco valorizado em países asiáticos. Suas dimensões variam de dois a trinta centímetros."

#### Atributos do dataset:
1. **Sex**: M, F e I (infantil)
2. **Length**: maior medida em mm da concha
3. **Diameter**: diâmetro em mm perpendicular a medida Length
4. **Height**: altura em mm com a carne dentro da concha
5. **Whole weight**: peso em gramas de toda a abalone
6. **Shucked weight**: peso em gramas da carne
7. **Viscera weight**: peso em gramas das vísceras após escorrer
8. **Shell weight**: peso em gramas para a concha após estar seca
9. **Type**: variável de classe (1, 2 ou 3) para o abalone

## Como usar (CSV)

Na pasta `03_Validation/csv`, há um script `abalone_validation_csv.py` que:

- Lê `abalone_dataset.csv` e `abalone_app.csv`;
- Monta um Pipeline com imputação (faltantes), normalização e one-hot para categóricos;
- Faz busca em grade para k no k-NN com validação cruzada estratificada repetida;
- Salva resultados completos em `knn_grid_results.csv` e previsões em `predictions.csv`/`predictions.json`;
- Opcionalmente envia o melhor resultado ao servidor.

Passos:

1. Instale dependências:

```bash
pip install -r requirements.txt
```

2. Execute a validação (exemplo com busca ampla):

```bash
python csv/abalone_validation_csv.py --min-k 1 --max-k 75 --step 2 --cv-splits 10 --cv-repeats 3
```

3. Envie as previsões do melhor modelo (opcional):

```bash
python csv/abalone_validation_csv.py --send --dev-key SUA_CHAVE_AQUI
```

Parâmetros úteis:

- `--dataset` e `--app` para caminhos customizados;
- `--min-k`, `--max-k`, `--step` para o espaço de busca;
- `--cv-splits`, `--cv-repeats`, `--random-state` para controle da validação;
- `--send` e `--dev-key` para envio ao servidor.

Observação (Windows): se ocorrer erro de multiprocessing, rode com `--n-jobs 1`.

Workflow recomendado para envio posterior:

1) Rode a busca e salve a melhor config (gera `best_params.json`):

```bash
python csv/abalone_validation_csv.py --min-k 1 --max-k 75 --step 2 --cv-splits 10 --cv-repeats 3 --n-jobs 1
```

2) Em outro momento, apenas gere as predições com a melhor config salva (pula a busca):

```bash
python csv/abalone_validation_csv.py --use-best --n-jobs 1
```

3) Opcionalmente, envie ao servidor:

```bash
python csv/abalone_validation_csv.py --use-best --send --dev-key SUA_CHAVE_AQUI --n-jobs 1
```
