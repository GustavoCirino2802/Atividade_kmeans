"""Configurações compartilhadas entre o módulo de treinamento e o de inferência."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Caminho padrão da base (coloque aqui o CSV disponibilizado na aula)
CAMINHO_BASE = BASE_DIR / "data" / "HousingData.csv"

# Onde o modelo treinado e os artefatos de normalização são salvos
CAMINHO_MODELO = BASE_DIR / "modelo" / "kmeans_boston.joblib"
CAMINHO_GRAFICO_COTOVELO = BASE_DIR / "modelo" / "cotovelo.png"

# Atributos categóricos da Boston Housing:
#   CHAS -> variável dummy (1 se o terreno faz divisa com o rio Charles, 0 caso contrário)
#   RAD  -> índice de acessibilidade a rodovias radiais (código discreto: 1..8, 24)
#   CAT. MEDV -> presente em algumas versões da base (1 se MEDV > 30, 0 caso contrário)
# Todos os demais atributos são tratados como quantitativos.
COLUNAS_CATEGORICAS = ["CHAS", "RAD", "CAT. MEDV"]

# Faixa de k avaliada no método do cotovelo
K_MIN = 1
K_MAX = 10

SEMENTE = 42
