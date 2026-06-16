"""Coletor de dados UNIFICADO (StatsBomb + FBref) → grava tudo em `data/raw/`.

Substitui o antigo `main.py` da raiz: baixa todas as fontes num lugar só, já no
formato/local que o pipeline (`football_analysis/main.py`) consome. Assim não há
mais a pasta `dados/` nem o passo de cópia.

O que coleta
------------
1) **StatsBomb Open Data** (`statsbombpy`, grátis, sem rate-limit) → CSVs:
     data/raw/sb_competicoes.csv, sb_partidas.csv, sb_escalacoes.csv,
     data/raw/sb_stats_jogador.csv   (agrega eventos por jogador/jogo, inclui xG)
2) **FBref por temporada** (`soccerdata`) → CSVs anuais por jogador:
     data/raw/fbref_jogadores_temporada/chutes/diversos/minutos.csv, fbref_goleiros.csv
3) **FBref por PARTIDA — ligas extras** (configs/fbref_sources.yaml) → parquet:
     data/raw/fbref/{schedule,player_match_summary,lineup}/   (Brasileirão, 2ª
     divisões, Argentina, Libertadores, etc. — match-level p/ as features de 18m)

Uso
---
    cd football_analysis
    python -m etl.sources.fetch_fbref                  # tudo
    python -m etl.sources.fetch_fbref --skip-statsbomb # pula a parte 1
    python -m etl.sources.fetch_fbref --skip-fbref-temporada
    python -m etl.sources.fetch_fbref --skip-fbref-partidas
    python -m etl.sources.fetch_fbref --max-competicoes 2   # teste rápido (parte 1)
    python -m etl.sources.fetch_fbref --only-extras "BRA-Serie A"

Requisitos:  pip install statsbombpy soccerdata pandas
Observação:  os imports de statsbombpy/soccerdata são LAZY (dentro das funções),
para este módulo poder ser importado sem puxar dependências pesadas.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]          # .../football_analysis
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_SOURCES_CONFIG = _PROJECT_ROOT / "configs" / "fbref_sources.yaml"

# Coletamos tudo cuja temporada termine em >= este ano.
ANO_MINIMO = 2010


# ======================================================================
# Utilidades
# ======================================================================
def _ano_final_temporada(season_name) -> int:
    """Maior ano de 4 dígitos no nome da temporada ('2023/2024' → 2024)."""
    anos = re.findall(r"\d{4}", str(season_name))
    return max(int(a) for a in anos) if anos else 0


def _load_sources_config() -> dict:
    with open(_SOURCES_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ======================================================================
# PARTE 1 — StatsBomb Open Data
# ======================================================================
def _agregar_stats_jogador(eventos):
    """Agrega as estatísticas por jogador num jogo (1 linha por jogador).

    Porta fiel do coletor original: contagem de cada tipo de evento (ev_*),
    métricas de chute (xG/gols/chutes no alvo) e de passe (completos/assist/key).
    Mantém o MESMO schema de `sb_stats_jogador.csv` que o pipeline já consome.
    """
    ev = eventos[eventos["player"].notna()].copy()
    if ev.empty:
        return None
    base = ev.groupby("player").agg(team=("team", "first"))

    contagem = ev.pivot_table(
        index="player", columns="type", values="id", aggfunc="count", fill_value=0
    )
    contagem.columns = [f"ev_{str(c).lower().replace(' ', '_')}" for c in contagem.columns]
    stats = base.join(contagem)

    def _col(df, nome):
        return df[nome] if nome in df.columns else None

    chutes = ev[ev["type"] == "Shot"]
    if not chutes.empty:
        if "shot_statsbomb_xg" in chutes.columns:
            stats = stats.join(chutes.groupby("player").agg(xg_total=("shot_statsbomb_xg", "sum")))
        if "shot_outcome" in chutes.columns:
            stats = stats.join(
                chutes[chutes["shot_outcome"] == "Goal"].groupby("player")["id"].count().rename("gols")
            ).join(
                chutes[chutes["shot_outcome"].isin(["Goal", "Saved"])]
                .groupby("player")["id"].count().rename("chutes_no_alvo")
            )

    passes = ev[ev["type"] == "Pass"]
    if not passes.empty:
        if "pass_outcome" in passes.columns:
            stats = stats.join(
                passes[passes["pass_outcome"].isna()]
                .groupby("player")["id"].count().rename("passes_completos")
            )
        for col, nome in [("pass_goal_assist", "assistencias"), ("pass_shot_assist", "key_passes")]:
            serie = _col(passes, col)
            if serie is not None:
                stats = stats.join(
                    passes[serie == True].groupby("player")["id"].count().rename(nome)  # noqa: E712
                )

    return stats.fillna(0).reset_index()


def coletar_statsbomb(out_dir: Path, max_competicoes: int | None = None) -> None:
    """Baixa partidas, escalações e stats por jogador do StatsBomb Open Data."""
    import pandas as pd
    from statsbombpy import sb

    out_dir.mkdir(parents=True, exist_ok=True)

    competicoes = sb.competitions()
    competicoes.to_csv(out_dir / "sb_competicoes.csv", index=False)
    print(f"[StatsBomb] {len(competicoes)} competições no catálogo.")

    competicoes = competicoes.copy()
    competicoes["ano_final"] = competicoes["season_name"].map(_ano_final_temporada)
    alvo = (
        competicoes[competicoes["ano_final"] >= ANO_MINIMO]
        .drop_duplicates(subset=["competition_id", "season_id"])
    )
    if max_competicoes:                       # atalho para teste rápido
        alvo = alvo.head(max_competicoes)
    print(f"[StatsBomb] {len(alvo)} competição/temporada desde {ANO_MINIMO}.")

    partidas_all, escal_all, stats_all = [], [], []

    def _salvar_parcial():
        if partidas_all:
            pd.concat(partidas_all, ignore_index=True).to_csv(out_dir / "sb_partidas.csv", index=False)
        if escal_all:
            pd.concat(escal_all, ignore_index=True).to_csv(out_dir / "sb_escalacoes.csv", index=False)
        if stats_all:
            pd.concat(stats_all, ignore_index=True).to_csv(out_dir / "sb_stats_jogador.csv", index=False)

    for i, (_, comp) in enumerate(alvo.iterrows(), 1):
        cid, sid = int(comp["competition_id"]), int(comp["season_id"])
        rotulo = f"{comp['competition_name']} {comp['season_name']}"
        try:
            partidas = sb.matches(competition_id=cid, season_id=sid)
        except Exception as e:
            print(f"  [!] partidas {rotulo}: {e}")
            continue
        if partidas is None or partidas.empty:
            continue
        partidas_all.append(partidas)
        print(f"  [{i}/{len(alvo)}] {rotulo}: {len(partidas)} partidas", flush=True)

        for match_id in partidas["match_id"]:
            try:
                for time, df_lineup in sb.lineups(match_id=match_id).items():
                    df_lineup = df_lineup.copy()
                    df_lineup["match_id"] = match_id
                    df_lineup["team"] = time
                    df_lineup["competition_id"] = cid
                    df_lineup["season_id"] = sid
                    escal_all.append(df_lineup)
                stats = _agregar_stats_jogador(sb.events(match_id=match_id))
                if stats is not None and not stats.empty:
                    stats["match_id"] = match_id
                    stats["competition_id"] = cid
                    stats["season_id"] = sid
                    stats_all.append(stats)
            except Exception as e:
                print(f"     [!] jogo {match_id}: {e}")
        _salvar_parcial()                     # progresso seguro por competição

    print(f"[StatsBomb] OK — CSVs em {out_dir}")


# ======================================================================
# PARTE 2 — FBref por TEMPORADA (agregados anuais por jogador)
# ======================================================================
def coletar_fbref_temporada(out_dir: Path) -> None:
    """Baixa os agregados de temporada do FBref (mesmos CSVs do coletor antigo)."""
    import time
    import pandas as pd
    import soccerdata as sd

    out_dir.mkdir(parents=True, exist_ok=True)
    # Injeta as ligas extras no league_dict para que entrem aqui também.
    _ensure_league_dict(_load_sources_config().get("competitions", []))

    ligas = [L for L in sd.FBref.available_leagues() if L != "Big 5 European Leagues Combined"]
    temporadas = [str(ano) for ano in range(ANO_MINIMO, date.today().year + 1)]
    print(f"[FBref/temporada] {len(ligas)} ligas, {len(temporadas)} temporadas.")

    tipos = {
        "standard": out_dir / "fbref_jogadores_temporada.csv",
        "shooting": out_dir / "fbref_jogadores_chutes.csv",
        "keeper": out_dir / "fbref_goleiros.csv",
        "playing_time": out_dir / "fbref_jogadores_minutos.csv",
        "misc": out_dir / "fbref_jogadores_diversos.csv",
    }
    acumulado = {t: [] for t in tipos}

    def _ler_retry(fbref, stat_type, tentativas=3):
        for k in range(1, tentativas + 1):
            try:
                return fbref.read_player_season_stats(stat_type=stat_type)
            except Exception:
                if k == tentativas:
                    raise
                time.sleep(15 * k)

    for liga in ligas:
        print(f"  -> {liga}", flush=True)
        for temp in temporadas:
            try:
                fbref = sd.FBref(leagues=liga, seasons=temp)
            except Exception as e:
                print(f"     [!] init {liga} {temp}: {str(e)[:70]}")
                continue
            for stat_type, caminho in tipos.items():
                try:
                    df = _ler_retry(fbref, stat_type)
                    if df is not None and not df.empty:
                        acumulado[stat_type].append(df)
                        pd.concat(acumulado[stat_type]).to_csv(caminho)
                except Exception as e:
                    print(f"     [!] {liga} {temp} '{stat_type}': {str(e)[:70]}")
    print(f"[FBref/temporada] OK — CSVs em {out_dir}")


# ======================================================================
# PARTE 3 — FBref por PARTIDA (ligas extras, match-level)
# ======================================================================
def _ensure_league_dict(competitions: list[dict]) -> None:
    """Injeta as competições extras no league_dict do soccerdata (idempotente)."""
    import json
    from soccerdata._config import CONFIG_DIR  # type: ignore

    path = Path(CONFIG_DIR) / "league_dict.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.load(path.open(encoding="utf-8")) if path.is_file() else {}
    for comp in competitions:
        entry = {"FBref": comp["fbref_name"]}
        if comp.get("single_year"):
            entry["season_code"] = "single-year"
        existing[comp["code"]] = {**existing.get(comp["code"], {}), **entry}
    json.dump(existing, path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _flatten_columns(df):
    import pandas as pd
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(p) for p in tup if str(p) not in ("", "nan")).strip("_") for tup in df.columns
        ]
    return df


def coletar_fbref_partidas(out_dir: Path, only: list[str] | None = None) -> None:
    """Baixa schedule + stats por jogador/partida + lineups das ligas extras."""
    import pandas as pd
    import soccerdata as sd

    cfg = _load_sources_config()
    seasons = [str(s) for s in cfg.get("seasons", [])]
    competitions = cfg.get("competitions", [])
    if only:
        competitions = [c for c in competitions if c["code"] in set(only)]
    fbref_dir = out_dir / "fbref"

    _ensure_league_dict(competitions)
    schedules, summaries, lineups = [], [], []
    for comp in competitions:
        code = comp["code"]
        print(f"\n[FBref/partida] {code} ({comp['fbref_name']}) {seasons}", flush=True)
        try:
            fb = sd.FBref(leagues=[code], seasons=seasons)
            summary = fb.read_player_match_stats(stat_type="summary").reset_index()
        except Exception as exc:
            print(f"[FBref/partida] PULANDO {code}: {exc!r}")
            continue
        if summary is None or len(summary) == 0:
            print(f"[FBref/partida] PULANDO {code}: sem stats por jogador.")
            continue
        summary = _flatten_columns(summary); summary["competition_code"] = code
        summaries.append(summary)
        for reader, bucket, nome in (
            (lambda: fb.read_schedule(), schedules, "schedule"),
            (lambda: fb.read_lineup(), lineups, "lineup"),
        ):
            try:
                df = _flatten_columns(reader().reset_index()); df["competition_code"] = code
                bucket.append(df)
            except Exception as exc:
                print(f"[FBref/partida] aviso {nome} {code}: {exc!r}")
        print(f"[FBref/partida] OK {code}: {len(summary)} linhas de stats por jogador.")

    def _save(frames, name):
        if not frames:
            return
        dest = fbref_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(dest / "part-0.parquet", index=False)
        print(f"[FBref/partida] gravado {name} em {dest}")

    _save(summaries, "player_match_summary")
    _save(schedules, "schedule")
    _save(lineups, "lineup")


# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Coletor unificado (StatsBomb + FBref) → data/raw/.")
    parser.add_argument("--skip-statsbomb", action="store_true")
    parser.add_argument("--skip-fbref-temporada", action="store_true")
    parser.add_argument("--skip-fbref-partidas", action="store_true")
    parser.add_argument("--max-competicoes", type=int, default=None,
                        help="Limita nº de competições do StatsBomb (teste rápido).")
    parser.add_argument("--only-extras", nargs="*", default=None,
                        help="Subconjunto de códigos de liga extra (match-level).")
    parser.add_argument("--out", default=str(_RAW_DIR), help="Diretório de saída (default data/raw).")
    args = parser.parse_args()
    out_dir = Path(args.out)

    if not args.skip_statsbomb:
        print(">>> StatsBomb Open Data ...")
        coletar_statsbomb(out_dir, max_competicoes=args.max_competicoes)
    if not args.skip_fbref_temporada:
        print("\n>>> FBref (temporada) ...")
        coletar_fbref_temporada(out_dir)
    if not args.skip_fbref_partidas:
        print("\n>>> FBref (por partida, ligas extras) ...")
        coletar_fbref_partidas(out_dir, only=args.only_extras)
    print(f"\nPronto! Dados em {out_dir}. Agora rode: python main.py")


if __name__ == "__main__":
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    main()
