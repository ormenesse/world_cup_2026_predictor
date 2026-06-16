#!/usr/bin/env python3
"""fetch_fifa_matches.py
------------------------
Baixa os SCHEDULES (datas + placar) das competições de `configs/fifa_match_sources.yaml`
via **FBref/soccerdata** usando apenas a API HTTP (sem Selenium) e grava tudo num
único parquet: ``data/raw/fifa_matches/schedule.parquet``.

Esses resultados são depois cruzados (pelo pipeline FIFA bronze/silver/gold,
orquestrado por ``configs/etl_config.yaml``) com os ratings de jogador do FIFA.
Só precisamos do schedule — os "jogadores
principais" de cada time vêm do próprio FIFA — então NÃO baixamos escalações nem
eventos por partida (1 requisição por liga-temporada → rápido e leve).

USO
---
    cd football_analysis
    python -m etl.sources.fetch_fifa_matches
    python -m etl.sources.fetch_fifa_matches --only "ITA-Serie A" "BRA-Serie A"
    python -m etl.sources.fetch_fifa_matches --force-cache   # usa só o cache local

Requisitos:  pip install soccerdata pandas pyyaml pyarrow
soccerdata faz scraping HTTP do FBref com rate-limit (~1 req/3s) e cacheia em
``~/soccerdata/``. É idempotente: relê do cache nas próximas execuções.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]            # .../football_analysis
_SOURCES_CONFIG = _PROJECT_ROOT / "configs" / "fifa_match_sources.yaml"


def _load_config() -> dict:
    with open(_SOURCES_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _ensure_league_dict(competitions: list[dict]) -> None:
    """Injeta competições que não vêm por padrão no soccerdata (ex.: amistosos).

    Só age para entradas que trazem `fbref_name` na config (idempotente).
    """
    import json
    from soccerdata._config import CONFIG_DIR  # type: ignore

    path = Path(CONFIG_DIR) / "league_dict.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.load(path.open(encoding="utf-8")) if path.is_file() else {}
    changed = False
    for comp in competitions:
        if not comp.get("fbref_name"):
            continue
        entry = {"FBref": comp["fbref_name"]}
        if comp.get("single_year"):
            entry["season_code"] = "single-year"
        merged = {**existing.get(comp["code"], {}), **entry}
        if existing.get(comp["code"]) != merged:
            existing[comp["code"]] = merged
            changed = True
    if changed:
        json.dump(existing, path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _parse_score(value) -> tuple[float, float]:
    """Extrai (gols_casa, gols_fora) de um placar tipo '2–1' / '2-1' / '2:1'.

    Ignora desempate por pênaltis entre parênteses (ex.: '1 (4)–1 (3)' → 1, 1).
    Retorna (nan, nan) se não houver placar (jogo não disputado).
    """
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan"), float("nan")
    s = str(value)
    nums = re.findall(r"(\d+)\s*(?:\(\d+\))?\s*[–\-:]\s*(\d+)", s)
    if not nums:
        return float("nan"), float("nan")
    h, a = nums[0]
    return float(h), float(a)


def _normalize_schedule(df, comp: dict):
    """Padroniza o schedule do FBref para o schema do dataset (1 linha/partida)."""
    import pandas as pd

    df = df.reset_index()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    date_c = pick("date")
    home_c = pick("home_team", "home")
    away_c = pick("away_team", "away")
    hs_c = pick("home_score", "home_goals")
    as_c = pick("away_score", "away_goals")
    score_c = pick("score")

    out = pd.DataFrame()
    out["match_date"] = pd.to_datetime(df[date_c], errors="coerce") if date_c else pd.NaT
    out["home_team"] = df[home_c].astype("string") if home_c else pd.NA
    out["away_team"] = df[away_c].astype("string") if away_c else pd.NA

    if hs_c and as_c:
        out["home_goals"] = pd.to_numeric(df[hs_c], errors="coerce")
        out["away_goals"] = pd.to_numeric(df[as_c], errors="coerce")
    elif score_c:
        parsed = df[score_c].map(_parse_score)
        out["home_goals"] = [p[0] for p in parsed]
        out["away_goals"] = [p[1] for p in parsed]
    else:
        out["home_goals"] = float("nan")
        out["away_goals"] = float("nan")

    out["competition_code"] = comp["code"]
    out["competition_group"] = comp["group"]
    out["match_type"] = comp["type"]
    return out


def fetch(only: list[str] | None = None, force_cache: bool = False) -> None:
    import pandas as pd
    import soccerdata as sd

    cfg = _load_config()
    competitions = cfg.get("competitions", [])
    if only:
        competitions = [c for c in competitions if c["code"] in set(only)]
    out_dir = _PROJECT_ROOT / cfg.get("raw_output_dir", "data/raw/fifa_matches")
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_league_dict(competitions)

    frames = []
    for comp in competitions:
        code, seasons = comp["code"], [str(s) for s in comp.get("seasons", [])]
        print(f"\n[schedule] {code}  seasons={seasons}", flush=True)
        for season in seasons:
            try:
                fb = sd.FBref(leagues=[code], seasons=[season])
                sched = fb.read_schedule(force_cache=force_cache)
            except Exception as exc:
                print(f"  [!] PULANDO {code} {season}: {str(exc)[:120]}")
                continue
            if sched is None or len(sched) == 0:
                print(f"  [-] {code} {season}: schedule vazio.")
                continue
            norm = _normalize_schedule(sched, comp)
            norm["season"] = season
            played = norm["home_goals"].notna().sum()
            frames.append(norm)
            print(f"  [+] {code} {season}: {len(norm)} jogos ({int(played)} com placar)")

    if not frames:
        print("\nNada baixado. Verifique rede/cache do soccerdata e os códigos de liga.",
              file=sys.stderr)
        sys.exit(1)

    full = pd.concat(frames, ignore_index=True)
    # dedup defensivo (mesma partida pode aparecer 2x se temporadas se sobrepuserem)
    full = full.drop_duplicates(
        subset=["competition_code", "match_date", "home_team", "away_team"]
    ).reset_index(drop=True)

    dest = out_dir / "schedule.parquet"
    full.to_parquet(dest, index=False)
    print(f"\n=== Resumo ===")
    print(f"  Partidas: {len(full):,}  (com placar: {full['home_goals'].notna().sum():,})")
    print(full.groupby("competition_group").size().to_string())
    print(f"  Salvo em: {dest}")


# --------------------------------------------------------------------------- #
# Alternativa SEM REDE: schedule a partir do StatsBomb já baixado.
# Útil quando o FBref está bloqueando (Cloudflare/CAPTCHA). Cobre só as
# competições que o StatsBomb Open Data tem (Copa do Mundo, Itália Serie A,
# Espanha La Liga, Alemanha 1. Bundesliga) — as 2ªs divisões / Brasil / amistosos
# continuam dependendo do FBref.
# --------------------------------------------------------------------------- #
# competição do StatsBomb -> (code da config, group, type)
_SB_COMP_MAP = {
    "International - FIFA World Cup": ("INT-World Cup", "world_cup", "nation"),
    "Italy - Serie A":               ("ITA-Serie A", "italy_a", "club"),
    "Spain - La Liga":               ("ESP-La Liga", "spain_a", "club"),
    "Germany - 1. Bundesliga":       ("GER-Bundesliga", "germany_a", "club"),
}


def fetch_from_statsbomb() -> None:
    """Monta data/raw/fifa_matches/schedule.parquet a partir de data/raw/sb_partidas.csv."""
    import pandas as pd

    cfg = _load_config()
    out_dir = _PROJECT_ROOT / cfg.get("raw_output_dir", "data/raw/fifa_matches")
    out_dir.mkdir(parents=True, exist_ok=True)
    min_date = pd.Timestamp(cfg.get("min_match_date", "2015-01-01"))

    src = _PROJECT_ROOT / "data" / "raw" / "sb_partidas.csv"
    df = pd.read_csv(src, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df[df["competition"].isin(_SB_COMP_MAP)].copy()
    if "home_team_gender" in df.columns:                 # só masculino
        df = df[df["home_team_gender"].fillna("male") == "male"]
    df = df[df["match_date"].notna() & (df["match_date"] >= min_date)]
    df = df[df["home_score"].notna() & df["away_score"].notna()]

    mapped = df["competition"].map(_SB_COMP_MAP)
    out = pd.DataFrame({
        "match_date": df["match_date"].values,
        "home_team": df["home_team"].astype("string").values,
        "away_team": df["away_team"].astype("string").values,
        "home_goals": pd.to_numeric(df["home_score"], errors="coerce").values,
        "away_goals": pd.to_numeric(df["away_score"], errors="coerce").values,
        "competition_code": [m[0] for m in mapped],
        "competition_group": [m[1] for m in mapped],
        "match_type": [m[2] for m in mapped],
        "season": df["season"].astype("string").values,
    }).drop_duplicates(subset=["competition_code", "match_date", "home_team", "away_team"])

    dest = out_dir / "schedule.parquet"
    out.to_parquet(dest, index=False)
    print(f"[statsbomb] {len(out):,} partidas (2015+, masc.) → {dest}")
    print(out.groupby("competition_group").size().to_string())


# --------------------------------------------------------------------------- #
# Alternativa ao FBref: CSVs ABERTOS de download direto (HTTP GET, sem scraping
# nem Selenium nem Cloudflare). Cobre TODOS os 7 grupos pedidos:
#   • football-data.co.uk  → Itália A/B, Espanha A/B, Alemanha A/B (mmz4281) e
#                            Brasil Série A (new/BRA.csv).
#   • martj42/international_results (GitHub raw) → Copa do Mundo e amistosos.
# --------------------------------------------------------------------------- #
_INTL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
_BRA_URL = "https://www.football-data.co.uk/new/BRA.csv"
_FD_BASE = "https://www.football-data.co.uk/mmz4281"

# div do football-data → (code da config, group)
_FD_DIVS = {
    "I1": ("ITA-Serie A", "italy_a"), "I2": ("ITA-Serie B", "italy_b"),
    "SP1": ("ESP-La Liga", "spain_a"), "SP2": ("ESP-Segunda", "spain_b"),
    "D1": ("GER-Bundesliga", "germany_a"), "D2": ("GER-2. Bundesliga", "germany_b"),
}
# tournament do dataset internacional → (code, group)
_INTL_TOURNAMENTS = {
    "FIFA World Cup": ("INT-World Cup", "world_cup"),
    "Friendly": ("INT-Friendlies", "national_friendly"),
}


def _http_csv(url: str, encoding: str = "latin-1"):
    """GET de um CSV com User-Agent → DataFrame (urllib; não depende de scraping)."""
    import io
    import urllib.request
    import pandas as pd

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return pd.read_csv(io.BytesIO(raw), encoding=encoding, on_bad_lines="skip", low_memory=False)


def fetch_from_open_csv(start_year: int = 2015) -> None:
    """Monta schedule.parquet a partir dos CSVs abertos (cobre os 7 grupos)."""
    import pandas as pd

    cfg = _load_config()
    out_dir = _PROJECT_ROOT / cfg.get("raw_output_dir", "data/raw/fifa_matches")
    out_dir.mkdir(parents=True, exist_ok=True)
    min_date = pd.Timestamp(cfg.get("min_match_date", "2015-01-01"))
    frames = []

    def _emit(df, code, group, mtype, season):
        df = df.copy()
        df["competition_code"] = code
        df["competition_group"] = group
        df["match_type"] = mtype
        df["season"] = str(season)
        frames.append(df[["match_date", "home_team", "away_team", "home_goals",
                           "away_goals", "competition_code", "competition_group",
                           "match_type", "season"]])

    # 1) Ligas de clube (football-data.co.uk, temporada cruzada)
    seasons = {y: f"{str(y)[2:]}{str(y + 1)[2:]}" for y in range(start_year, 2025)}
    for div, (code, group) in _FD_DIVS.items():
        for y, scode in seasons.items():
            url = f"{_FD_BASE}/{scode}/{div}.csv"
            try:
                raw = _http_csv(url)
            except Exception as exc:
                print(f"  [!] {code} {scode}: {str(exc)[:80]}")
                continue
            cols = {c.lower(): c for c in raw.columns}
            need = [cols.get(k) for k in ("date", "hometeam", "awayteam", "fthg", "ftag")]
            if any(c is None for c in need):
                continue
            d, h, a, hg, ag = need
            std = pd.DataFrame({
                "match_date": pd.to_datetime(raw[d], dayfirst=True, errors="coerce"),
                "home_team": raw[h].astype("string"),
                "away_team": raw[a].astype("string"),
                "home_goals": pd.to_numeric(raw[hg], errors="coerce"),
                "away_goals": pd.to_numeric(raw[ag], errors="coerce"),
            }).dropna(subset=["match_date", "home_goals", "away_goals"])
            _emit(std, code, group, "club", f"{y}-{y + 1}")
            print(f"  [+] {code} {y}-{y+1}: {len(std)} jogos")

    # 2) Brasil Série A (football-data.co.uk new — arquivo único, BOM)
    try:
        bra = _http_csv(_BRA_URL, encoding="utf-8-sig")
        bra = bra[bra["League"].astype(str).str.strip() == "Serie A"]
        std = pd.DataFrame({
            "match_date": pd.to_datetime(bra["Date"], dayfirst=True, errors="coerce"),
            "home_team": bra["Home"].astype("string"),
            "away_team": bra["Away"].astype("string"),
            "home_goals": pd.to_numeric(bra["HG"], errors="coerce"),
            "away_goals": pd.to_numeric(bra["AG"], errors="coerce"),
            "season": bra["Season"].astype("string"),
        }).dropna(subset=["match_date", "home_goals", "away_goals"])
        std = std[std["match_date"] >= min_date]
        for season, g in std.groupby("season"):
            _emit(g, "BRA-Serie A", "brazil_a", "club", season)
        print(f"  [+] BRA-Serie A: {len(std)} jogos (2015+)")
    except Exception as exc:
        print(f"  [!] BRA-Serie A: {str(exc)[:80]}")

    # 3) Internacional: Copa do Mundo + amistosos (GitHub raw)
    try:
        intl = _http_csv(_INTL_URL, encoding="utf-8")
        intl["match_date"] = pd.to_datetime(intl["date"], errors="coerce")
        intl = intl[intl["match_date"] >= min_date]
        for tourn, (code, group) in _INTL_TOURNAMENTS.items():
            t = intl[intl["tournament"] == tourn]
            std = pd.DataFrame({
                "match_date": t["match_date"],
                "home_team": t["home_team"].astype("string"),
                "away_team": t["away_team"].astype("string"),
                "home_goals": pd.to_numeric(t["home_score"], errors="coerce"),
                "away_goals": pd.to_numeric(t["away_score"], errors="coerce"),
            }).dropna(subset=["home_goals", "away_goals"])
            std["year"] = std["match_date"].dt.year
            for yr, g in std.groupby("year"):
                _emit(g.drop(columns="year"), code, group, "nation", yr)
            print(f"  [+] {code}: {len(std)} jogos (2015+)")
    except Exception as exc:
        print(f"  [!] internacional: {str(exc)[:80]}")

    if not frames:
        print("\nNada baixado dos CSVs abertos.", file=sys.stderr)
        sys.exit(1)

    full = pd.concat(frames, ignore_index=True)
    full = full[full["match_date"] >= min_date]
    full["home_goals"] = full["home_goals"].astype(int)
    full["away_goals"] = full["away_goals"].astype(int)
    full = full.drop_duplicates(
        subset=["competition_code", "match_date", "home_team", "away_team"]
    ).reset_index(drop=True)

    dest = out_dir / "schedule.parquet"
    full.to_parquet(dest, index=False)
    print(f"\n=== Resumo (CSVs abertos) ===\n  Partidas: {len(full):,} → {dest}")
    print(full.groupby("competition_group").size().to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Baixa schedules p/ cruzar com FIFA.")
    ap.add_argument("--only", nargs="*", default=None, help="Subconjunto de codes de liga (FBref).")
    ap.add_argument("--force-cache", action="store_true",
                    help="Usa só o cache local do soccerdata (não vai à rede).")
    ap.add_argument("--from-statsbomb", action="store_true",
                    help="NÃO usa o FBref; monta o schedule a partir de data/raw/sb_partidas.csv "
                         "(cobre Copa do Mundo, Itália A, Espanha A, Alemanha A).")
    ap.add_argument("--from-open-csv", action="store_true",
                    help="NÃO usa o FBref; baixa CSVs abertos (football-data.co.uk + "
                         "martj42/international_results). Cobre os 7 grupos pedidos.")
    args = ap.parse_args()
    if args.from_open_csv:
        fetch_from_open_csv()
    elif args.from_statsbomb:
        fetch_from_statsbomb()
    else:
        fetch(only=args.only, force_cache=args.force_cache)


if __name__ == "__main__":
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    main()
