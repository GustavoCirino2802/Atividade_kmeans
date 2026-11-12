"""
Módulo de treinamento - Clusterização da base Boston Housing com K-Means.

Etapas:
  1. Carrega a base e trata valores ausentes.
  2. Normaliza separadamente:
       - atributos quantitativos  -> MinMaxScaler (escala 0..1)
       - atributos categóricos    -> One-Hot Encoding (0/1)
     (não há balanceamento: clusterização é não supervisionada)
  3. Determina o número ótimo de grupos pelo método do cotovelo, usando a
     inércia (soma das distâncias quadráticas de cada ponto ao seu centroide).
     O "cotovelo" é escolhido automaticamente como o ponto da curva com maior
     distância perpendicular à reta que liga o primeiro e o último ponto.
  4. Treina o K-Means final e salva modelo + normalizadores em disco.

Uso:
    python treinamento.py                      # usa data/HousingData.csv
    python treinamento.py --base caminho.csv   # outra base
    python treinamento.py --k 4                # força um k específico
"""
import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import config


# ---------------------------------------------------------------------------
# 1. Carga e limpeza
# ---------------------------------------------------------------------------
def carregar_base(caminho):
    """Lê o CSV (vírgula ou ponto e vírgula) e padroniza os nomes das colunas."""
    df = pd.read_csv(caminho, sep=None, engine="python")
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("UNNAMED")]  # índice salvo por engano
    df = df.dropna(axis=1, how="all")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def separar_colunas(df):
    categoricas = [c for c in config.COLUNAS_CATEGORICAS if c in df.columns]
    quantitativas = [c for c in df.columns if c not in categoricas]
    return quantitativas, categoricas


def tratar_ausentes(df, quantitativas, categoricas):
    """Quantitativos: mediana. Categóricos: moda. Retorna os valores usados."""
    preenchimento = {}
    for c in quantitativas:
        preenchimento[c] = float(df[c].median())
    for c in categoricas:
        preenchimento[c] = df[c].mode().iloc[0]
    n_ausentes = int(df.isna().sum().sum())
    df = df.fillna(preenchimento)
    for c in categoricas:  # categorias como inteiros (ex.: 1.0 -> 1)
        df[c] = df[c].round().astype(int)
    return df, preenchimento, n_ausentes


# ---------------------------------------------------------------------------
# 2. Normalização
# ---------------------------------------------------------------------------
def ajustar_normalizadores(df, quantitativas, categoricas):
    scaler = MinMaxScaler().fit(df[quantitativas])
    encoder = None
    if categoricas:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(df[categoricas])
    return scaler, encoder


def normalizar(df, scaler, encoder, quantitativas, categoricas):
    partes = [scaler.transform(df[quantitativas])]
    if encoder is not None:
        partes.append(encoder.transform(df[categoricas]))
    return np.hstack(partes)


# ---------------------------------------------------------------------------
# 3. Método do cotovelo
# ---------------------------------------------------------------------------
def calcular_inercias(X, k_min, k_max):
    ks = list(range(k_min, k_max + 1))
    inercias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.SEMENTE).fit(X)
        inercias.append(km.inertia_)
        print(f"  k = {k:2d} | inércia (soma das distâncias²) = {km.inertia_:10.4f}")
    return ks, inercias


def k_otimo_cotovelo(ks, inercias):
    """Ponto da curva mais distante da reta entre o primeiro e o último ponto."""
    x = np.array(ks, dtype=float)
    y = np.array(inercias, dtype=float)
    # normaliza os eixos para que a escala não distorça a distância
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    direcao = (p2 - p1) / np.linalg.norm(p2 - p1)
    distancias = []
    for xi, yi in zip(xn, yn):
        v = np.array([xi, yi]) - p1
        distancias.append(abs(v[0] * direcao[1] - v[1] * direcao[0]))
    return ks[int(np.argmax(distancias))], distancias


def salvar_grafico(ks, inercias, k_escolhido, caminho):
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inercias, "o-", color="#1f77b4")
    idx = ks.index(k_escolhido)
    plt.plot(k_escolhido, inercias[idx], "o", color="#d62728", markersize=12,
             label=f"cotovelo: k = {k_escolhido}")
    plt.title("Método do cotovelo - K-Means / Boston Housing")
    plt.xlabel("Número de grupos (k)")
    plt.ylabel("Inércia (soma das distâncias² aos centroides)")
    plt.xticks(ks)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Treina K-Means na Boston Housing")
    parser.add_argument("--base", default=str(config.CAMINHO_BASE))
    parser.add_argument("--k", type=int, default=None,
                        help="força um número de grupos (ignora o cotovelo)")
    args = parser.parse_args()

    print(f"[1] Carregando base: {args.base}")
    df = carregar_base(args.base)
    quantitativas, categoricas = separar_colunas(df)
    df, preenchimento, n_ausentes = tratar_ausentes(df, quantitativas, categoricas)
    print(f"    {df.shape[0]} registros, {df.shape[1]} atributos, "
          f"{n_ausentes} valores ausentes preenchidos")
    print(f"    Quantitativos ({len(quantitativas)}): {quantitativas}")
    print(f"    Categóricos   ({len(categoricas)}): {categoricas}")

    print("[2] Normalizando (MinMax nos quantitativos, One-Hot nos categóricos)")
    scaler, encoder = ajustar_normalizadores(df, quantitativas, categoricas)
    X = normalizar(df, scaler, encoder, quantitativas, categoricas)
    print(f"    Matriz normalizada: {X.shape}")

    print(f"[3] Método do cotovelo (k = {config.K_MIN}..{config.K_MAX})")
    ks, inercias = calcular_inercias(X, config.K_MIN, config.K_MAX)
    k_cotovelo, _ = k_otimo_cotovelo(ks, inercias)
    k_final = args.k or k_cotovelo
    print(f"    Cotovelo identificado em k = {k_cotovelo}"
          + ("" if args.k is None else f" (usando k = {k_final} informado)"))
    config.CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    salvar_grafico(ks, inercias, k_cotovelo, config.CAMINHO_GRAFICO_COTOVELO)
    print(f"    Gráfico salvo em {config.CAMINHO_GRAFICO_COTOVELO}")

    print(f"[4] Treinando K-Means final com k = {k_final}")
    modelo = KMeans(n_clusters=k_final, n_init=10, random_state=config.SEMENTE).fit(X)
    tamanhos = np.bincount(modelo.labels_)
    for g, n in enumerate(tamanhos):
        print(f"    Grupo {g}: {n} residências")

    artefato = {
        "modelo": modelo,
        "scaler": scaler,
        "encoder": encoder,
        "quantitativas": quantitativas,
        "categoricas": categoricas,
        "preenchimento": preenchimento,
        "ordem_colunas": quantitativas + categoricas,
        "k": k_final,
        "inercias": dict(zip(ks, inercias)),
    }
    joblib.dump(artefato, config.CAMINHO_MODELO)
    print(f"    Modelo salvo em {config.CAMINHO_MODELO}")


if __name__ == "__main__":
    main()
