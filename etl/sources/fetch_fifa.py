#!/usr/bin/env python3
"""
aggregate_fifa.py
-----------------
BAIXA (via API do Kaggle) e empilha datasets de jogadores de varias edicoes
do FIFA / EA Sports FC (FIFA 15 ... FC 26) num unico CSV com historico ano a ano.

A maioria desses datasets do Kaggle deriva do sofifa.com e compartilha a mesma
estrutura de colunas, mas alguns usam nomes diferentes (ex.: 'sofifa_id' vs
'player_id', 'name' vs 'short_name'). O script normaliza os nomes, garante uma
coluna `fifa_version` em todos e concatena tudo.

-------------------------------------------------------------------------------
PRE-REQUISITOS (uma vez so):
  1. pip install kaggle pandas
  2. Gere seu token: kaggle.com -> Settings -> API -> "Create New Token"
     Isso baixa um kaggle.json. Coloque em:
        Linux/Mac:  ~/.kaggle/kaggle.json   (e: chmod 600 ~/.kaggle/kaggle.json)
        Windows:    C:\\Users\\<voce>\\.kaggle\\kaggle.json
     (ou exporte as variaveis KAGGLE_USERNAME e KAGGLE_KEY)

USO:
  python aggregate_fifa.py
  -> baixa os datasets em ./data/raw/<slug>/, monta ./data/fifa_aggregated.csv
  Se os arquivos ja estiverem baixados, ele NAO baixa de novo (idempotente).

  Para pular o download e usar so o que ja existe:  python aggregate_fifa.py --no-download
-------------------------------------------------------------------------------
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
import pandas as pd


# --------------------------------------------------------------------------- #
# 1) DATASETS: slug do Kaggle + qual versao atribuir.
#
#    - slug:    "<usuario>/<dataset>"  (o que aparece na URL do Kaggle)
#    - version: rotulo da edicao a aplicar SE o arquivo nao tiver 'fifa_version'
#               proprio. Os agregados historicos ja trazem a coluna -> None.
#    - include: padroes glob de quais CSVs usar dentro do dataset (player files)
#    - exclude: padroes a ignorar (times, tecnicos, feminino, legacy...)
# --------------------------------------------------------------------------- #
DATASETS = [
    {
        # Agregado historico: FIFA 15..23 (traz fifa_version interno)
        "slug": "stefanoleone992/fifa-23-complete-player-dataset",
        "version": None,
        "include": ["*male_players*.csv", "*players*.csv"],
        "exclude": ["*female*", "*team*", "*coach*", "*legacy*"],
    },
    {
        # FC 26 (homens / mulheres / combinado) - sem fifa_version interno
        "slug": "flynn28/eafc26-player-database",
        "version": 26,
        "include": ["*combined*.csv", "*men*.csv", "*player*.csv"],
        "exclude": ["*women*", "*female*"],
    },
    # Exemplo de como adicionar o FC 25 quando voce tiver o slug:
    # {
    #     "slug": "<usuario>/<dataset-fc25>",
    #     "version": 25,
    #     "include": ["*player*.csv"],
    #     "exclude": ["*team*", "*coach*"],
    # },
]

RAW_DIR = os.path.join("..", "..", "data", "raw")
OUTPUT_CSV = os.path.join("..", "..", "data", "fifa_aggregated.csv")


# --------------------------------------------------------------------------- #
# 2) MAPA DE ALIASES -> nome canonico. Adicione aqui se um dataset usar
#    nomes diferentes dos previstos.
# --------------------------------------------------------------------------- #
COLUMN_ALIASES = {
    "sofifa_id": "player_id", "player_id": "player_id", "id": "player_id",
    "short_name": "short_name", "name": "short_name",
    "long_name": "long_name", "full_name": "long_name",
    "overall": "overall", "ovr": "overall",
    "potential": "potential", "pot": "potential",
    "age": "age",
    "dob": "dob", "birth_date": "dob",
    "height_cm": "height_cm", "height": "height_cm",
    "weight_kg": "weight_kg", "weight": "weight_kg",
    "club_name": "club_name", "club": "club_name", "team": "club_name",
    "league_name": "league_name", "league": "league_name",
    "nationality_name": "nationality", "nationality": "nationality", "nation": "nationality",
    "player_positions": "positions", "positions": "positions", "position": "positions",
    "pace": "pace", "shooting": "shooting", "passing": "passing",
    "dribbling": "dribbling", "defending": "defending",
    "physic": "physic", "physicality": "physic", "physical": "physic",
    "value_eur": "value_eur", "value": "value_eur",
    "wage_eur": "wage_eur", "wage": "wage_eur",
    "fifa_version": "fifa_version", "fifa_update": "fifa_update",
}


def download_dataset(slug: str, dest_root: str) -> str:
    """Baixa e descompacta um dataset do Kaggle em dest_root/<slug-folder>.
    Retorna a pasta de destino. Pula se ja houver CSVs la."""
    folder = os.path.join(dest_root, slug.replace("/", "__"))
    os.makedirs(folder, exist_ok=True)

    if glob.glob(os.path.join(folder, "*.csv")):
        print(f"  = ja baixado: {slug}")
        return folder

    try:
        # import preguicoso: so exige o kaggle quando realmente vamos baixar
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print(f"  [ERRO] pacote 'kaggle' nao instalado. Rode: pip install kaggle",
              file=sys.stderr)
        return folder

    try:
        api = KaggleApi()
        api.authenticate()  # le ~/.kaggle/kaggle.json ou KAGGLE_USERNAME/KAGGLE_KEY
        print(f"  v baixando: {slug} ...")
        api.dataset_download_files(slug, path=folder, unzip=True, quiet=False)
    except Exception as exc:
        print(f"  [ERRO] falha ao baixar {slug}: {exc}", file=sys.stderr)
        print( "        Verifique seu kaggle.json (Settings -> API no Kaggle) "
               "e se aceitou as regras do dataset no site.", file=sys.stderr)
    return folder


def pick_csvs(folder: str, include: list[str], exclude: list[str]) -> list[str]:
    """Escolhe os CSVs de jogadores dentro da pasta do dataset."""
    found: list[str] = []
    for pat in include:
        found += glob.glob(os.path.join(folder, "**", pat), recursive=True)
    # dedup preservando ordem
    seen, ordered = set(), []
    for f in found:
        if f not in seen:
            seen.add(f); ordered.append(f)
    # aplica exclude
    def excluded(path: str) -> bool:
        base = os.path.basename(path).lower()
        return any(glob.fnmatch.fnmatch(base, pat.lower()) for pat in exclude)
    ordered = [f for f in ordered if not excluded(f)]
    # se nada bateu no include, cai pra qualquer csv nao-excluido
    if not ordered:
        allcsv = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
        ordered = [f for f in allcsv if not excluded(f)]
    return ordered


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename)


def load_csv(path: str, version) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    if "fifa_version" not in df.columns or df["fifa_version"].isna().all():
        if version is not None:
            df["fifa_version"] = version
        else:
            print(f"  [AVISO] {path} sem 'fifa_version' e sem version na config.",
                  file=sys.stderr)
    df["__source_file"] = os.path.basename(path)
    print(f"    + {os.path.basename(path)}: {len(df):,} linhas")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true",
                    help="nao baixa do Kaggle; usa apenas arquivos ja presentes")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    frames = []

    for ds in DATASETS:
        print(f"\nDataset: {ds['slug']}")
        if args.no_download:
            folder = os.path.join(RAW_DIR, ds["slug"].replace("/", "__"))
        else:
            folder = download_dataset(ds["slug"], RAW_DIR)

        csvs = pick_csvs(folder, ds.get("include", ["*.csv"]), ds.get("exclude", []))
        if not csvs:
            print(f"  [AVISO] nenhum CSV encontrado em {folder}", file=sys.stderr)
            continue
        for path in csvs:
            frames.append(load_csv(path, ds["version"]))

    if not frames:
        print("\nNada para agregar. Confira a autenticacao do Kaggle e os slugs.",
              file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    for c in ("fifa_version", "overall", "potential", "age"):
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c], errors="coerce")

    sort_cols = [c for c in ("player_id", "fifa_version") if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print("\n=== Resumo do agregado ===")
    print(f"  Linhas (jogador x edicao): {len(combined):,}")
    if "player_id" in combined.columns:
        print(f"  Jogadores unicos: {combined['player_id'].nunique():,}")
    if "fifa_version" in combined.columns:
        print(f"  Edicoes: {sorted(combined['fifa_version'].dropna().unique().tolist())}")
    print(f"  Salvo em: {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()