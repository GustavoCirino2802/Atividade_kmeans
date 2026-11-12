"""
Módulo de inferência - Em qual grupo uma nova residência se encaixa?

Etapas:
  1. Recebe os dados de uma nova residência (arquivo JSON, argumentos ou teclado).
  2. Normaliza com os MESMOS normalizadores ajustados no treinamento.
  3. Infere o grupo com o K-Means salvo.
  4. Exibe o centroide do grupo inferido desnormalizado (na escala original).

Uso:
    python inferencia.py --json nova_residencia.json
    python inferencia.py --valores CRIM=0.03 ZN=0 INDUS=7.07 CHAS=0 ...
    python inferencia.py                     # pergunta cada atributo no teclado
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

import config

DESCRICOES = {
    "CRIM": "taxa de criminalidade per capita",
    "ZN": "% de terrenos residenciais > 25.000 pés²",
    "INDUS": "% de área comercial não varejista",
    "CHAS": "faz divisa com o rio Charles (1 = sim, 0 = não)",
    "NOX": "concentração de óxidos nítricos (partes por 10 milhões)",
    "RM": "número médio de cômodos",
    "AGE": "% de imóveis construídos antes de 1940",
    "DIS": "distância ponderada a 5 centros de emprego",
    "RAD": "índice de acessibilidade a rodovias radiais",
    "TAX": "imposto sobre propriedade (por US$ 10.000)",
    "PTRATIO": "razão aluno/professor",
    "B": "1000(Bk - 0,63)², Bk = proporção de negros",
    "LSTAT": "% da população de baixa renda",
    "MEDV": "valor mediano do imóvel (US$ 1.000)",
    "CAT. MEDV": "1 se MEDV > 30, 0 caso contrário",
}


# ---------------------------------------------------------------------------
# 1. Entrada dos dados
# ---------------------------------------------------------------------------
def ler_entrada(args, colunas):
    if args.json:
        with open(args.json, encoding="utf-8") as f:
            dados = json.load(f)
    elif args.valores:
        dados = {}
        for item in args.valores:
            chave, valor = item.split("=", 1)
            dados[chave] = valor
    else:
        print("Informe os dados da nova residência (Enter = valor mediano/moda da base):")
        dados = {}
        for c in colunas:
            v = input(f"  {c:9s} - {DESCRICOES.get(c, '')}: ").strip()
            if v:
                dados[c] = v
    return {k.strip().upper(): v for k, v in dados.items()}


def montar_registro(dados, art):
    registro = {}
    faltantes = []
    for c in art["ordem_colunas"]:
        if c in dados and dados[c] not in ("", None):
            registro[c] = float(dados[c])
        else:
            registro[c] = art["preenchimento"][c]
            faltantes.append(c)
    if faltantes:
        print(f"Aviso: atributos não informados preenchidos com mediana/moda: {faltantes}")
    df = pd.DataFrame([registro])
    for c in art["categoricas"]:
        df[c] = df[c].round().astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Normalização
# ---------------------------------------------------------------------------
def normalizar(df, art):
    partes = [art["scaler"].transform(df[art["quantitativas"]])]
    if art["encoder"] is not None:
        enc = art["encoder"]
        for c, cats in zip(art["categoricas"], enc.categories_):
            if df[c].iloc[0] not in cats:
                print(f"Aviso: {c} = {df[c].iloc[0]} não existia no treino "
                      f"(categorias conhecidas: {list(cats)}); será ignorado.")
        partes.append(enc.transform(df[art["categoricas"]]))
    return np.hstack(partes)


# ---------------------------------------------------------------------------
# 4. Desnormalização do centroide
# ---------------------------------------------------------------------------
def desnormalizar_centroide(centroide, art):
    nq = len(art["quantitativas"])
    quant = art["scaler"].inverse_transform(centroide[:nq].reshape(1, -1))[0]
    resultado = {c: v for c, v in zip(art["quantitativas"], quant)}
    detalhes_cat = {}
    if art["encoder"] is not None:
        pos = nq
        for c, cats in zip(art["categoricas"], art["encoder"].categories_):
            bloco = centroide[pos:pos + len(cats)]
            pos += len(cats)
            # no centroide, o bloco one-hot vira a proporção de cada categoria
            # no grupo; a categoria desnormalizada é a predominante (maior proporção)
            resultado[c] = int(cats[int(np.argmax(bloco))])
            detalhes_cat[c] = {int(k): float(p) for k, p in zip(cats, bloco) if p > 0.005}
    return resultado, detalhes_cat


def main():
    parser = argparse.ArgumentParser(description="Inferência de grupo (K-Means)")
    parser.add_argument("--json", help="arquivo JSON com os dados da residência")
    parser.add_argument("--valores", nargs="*", help="pares ATRIBUTO=valor")
    parser.add_argument("--modelo", default=str(config.CAMINHO_MODELO))
    args = parser.parse_args()

    art = joblib.load(args.modelo)
    modelo = art["modelo"]

    dados = ler_entrada(args, art["ordem_colunas"])
    df = montar_registro(dados, art)
    X = normalizar(df, art)

    grupo = int(modelo.predict(X)[0])
    distancias = np.linalg.norm(modelo.cluster_centers_ - X, axis=1)
    centroide, detalhes_cat = desnormalizar_centroide(modelo.cluster_centers_[grupo], art)

    print("\n" + "=" * 72)
    print(f"Nova residência pertence ao GRUPO {grupo} (de {art['k']} grupos)")
    print(f"Distância ao centroide (espaço normalizado): {distancias[grupo]:.4f}")
    print("=" * 72)
    print(f"{'Atributo':10s} {'Nova residência':>16s} {'Centroide grupo ' + str(grupo):>20s}  Descrição")
    print("-" * 72)
    for c in art["ordem_colunas"]:
        print(f"{c:10s} {df[c].iloc[0]:16.4f} {centroide[c]:20.4f}  {DESCRICOES.get(c, '')}")
    if detalhes_cat:
        print("\nComposição das variáveis categóricas no grupo (proporção):")
        for c, props in detalhes_cat.items():
            texto = ", ".join(f"{k}: {p:.0%}" for k, p in props.items())
            print(f"  {c}: {texto}")


if __name__ == "__main__":
    main()
