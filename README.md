# Clusterização da Boston Housing com K-Means

Avaliação do 2º bimestre: clusterizador K-Means treinado na base **Boston Housing**, com módulo de treinamento e módulo de inferência.

## Estrutura

```
├── config.py              # caminhos e definição das colunas categóricas
├── treinamento.py         # módulo de treinamento
├── inferencia.py          # módulo de inferência
├── nova_residencia.json   # exemplo de nova residência
├── data/HousingData.csv   # base da aula
└── modelo/
    ├── kmeans_boston.joblib   # modelo + normalizadores salvos
    └── cotovelo.png           # gráfico do método do cotovelo
```

## Instalação

```bash
pip install -r requirements.txt
```

## Módulo de treinamento

```bash
python treinamento.py                       # usa data/HousingData.csv
python treinamento.py --base outra_base.csv # outra base
python treinamento.py --k 4                 # força um k específico
```

O que ele faz:

1. **Carga e limpeza**: lê o CSV, padroniza nomes de colunas e preenche valores ausentes (mediana para quantitativos, moda para categóricos).
2. **Normalização com cuidado sobre o tipo de dado**:
   - **Categóricos** (`CHAS`, `RAD` e, se existir, `CAT. MEDV`) → **One-Hot Encoding**. `CHAS` é dummy (rio Charles) e `RAD` é um código de acessibilidade (1–8, 24): a distância numérica entre os códigos não tem significado, então eles não podem ser escalados como números.
   - **Quantitativos** (demais atributos) → **MinMaxScaler** (0 a 1), deixando todos na mesma escala das colunas one-hot, para que nenhum atributo domine a distância euclidiana do K-Means.
3. **Sem balanceamento**: a clusterização é não supervisionada, então não há classes a balancear.
4. **Método do cotovelo (distâncias)**: treina o K-Means para k = 1…10 e registra a **inércia** (soma das distâncias quadráticas de cada ponto ao seu centroide). O cotovelo é escolhido automaticamente como o ponto da curva com **maior distância perpendicular à reta** que liga o primeiro e o último ponto. O gráfico é salvo em `modelo/cotovelo.png`.
5. **Treina e salva** o modelo final junto com os normalizadores (`modelo/kmeans_boston.joblib`).

## Módulo de inferência

```bash
python inferencia.py --json nova_residencia.json
python inferencia.py --valores CRIM=0.03 RM=6.4 CHAS=0 RAD=2 MEDV=21.6
python inferencia.py      # pede cada atributo pelo teclado
```

O que ele faz:

1. **Recebe** os dados da nova residência (JSON, argumentos ou teclado). Atributos não informados recebem a mediana/moda da base de treino.
2. **Normaliza** com os **mesmos** `MinMaxScaler` e `OneHotEncoder` ajustados no treino.
3. **Infere** o grupo com `modelo.predict`.
4. **Exibe o centroide desnormalizado**:
   - quantitativos: `scaler.inverse_transform` → valores na escala original;
   - categóricos: no centroide, cada bloco one-hot representa a proporção de cada categoria no grupo; a categoria desnormalizada é a predominante, e a composição completa também é exibida.

## Atributos da base

| Atributo | Tipo | Descrição |
|---|---|---|
| CRIM | quantitativo | taxa de criminalidade per capita |
| ZN | quantitativo | % de terrenos residenciais > 25.000 pés² |
| INDUS | quantitativo | % de área comercial não varejista |
| CHAS | **categórico** | 1 se faz divisa com o rio Charles |
| NOX | quantitativo | concentração de óxidos nítricos |
| RM | quantitativo | número médio de cômodos |
| AGE | quantitativo | % de imóveis construídos antes de 1940 |
| DIS | quantitativo | distância ponderada a centros de emprego |
| RAD | **categórico** | índice de acessibilidade a rodovias radiais |
| TAX | quantitativo | imposto sobre propriedade |
| PTRATIO | quantitativo | razão aluno/professor |
| B | quantitativo | 1000(Bk − 0,63)² |
| LSTAT | quantitativo | % da população de baixa renda |
| MEDV | quantitativo | valor mediano do imóvel (US$ 1.000) |
