"""Features de Copa do Mundo a partir da gold `gold_fifa_partidas`.

Variante de `world_cup_features.py` que monta as linhas de previsão a partir da
gold FIFA (`data/gold_fifa_partidas/`) em vez da gold de futebol StatsBomb/FBref.

Diferenças em relação ao original
---------------------------------
• Universo de features: na gold FIFA cada lado tem o SCORE AGREGADO do XI vigente
  do FIFA — `{lado}_xi_<attr>_mean` (overall, potential, value_eur, age, pace,
  shooting, passing, dribbling, defending, physic) e as médias por setor
  `{lado}_xi_overall_mean_{GK,DEF,MID,FWD}`. O filtro `"mean" in col` seleciona
  exatamente essas colunas (os 11 slots `{lado}_p01..p11_*` NÃO contêm "mean" e
  ficam de fora, como pretendido).
• Universo de times: usamos só as partidas de SELEÇÃO (`match_type == "nation"`),
  e o snapshot mais recente em que o time foi CASADO ao FIFA (`{lado}_matched`),
  evitando snapshots nulos de jogos sem elenco resolvido.
• Nomes: a gold FIFA usa os nomes do schedule (martj42/international_results),
  ex.: "United States", "South Korea", "Iran", "Czech Republic", "Cape Verde",
  "Ivory Coast", "DR Congo". O mapa de aliases abaixo converte os nomes das
  fixtures (FIFA/oficiais) para esses.

A mecânica (último snapshot por time → perspectiva home/away → fallback pela
mediana → linha alinhada às colunas de treino) é idêntica à do arquivo original,
então a API pública (`build_team_feature_store`, `prepare_match_row`,
`prepare_world_cup_matches`) é a mesma.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Fixture (nome oficial/FIFA) → nome do time na gold FIFA (international_results).
# Times não listados casam pelo mesmo nome (ex.: "Brazil", "Spain", "Croatia").
TEAM_NAME_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
}

# --------------------------------------------------------------------------- #
# Construção de features a partir de uma ESCALAÇÃO custom (usado pelo app).
# Os nomes/fórmulas espelham EXATAMENTE o job gold `etl/2_gold/gold_fifa_partidas.py`
# (mantidos aqui em python para o app não depender do polars/macros).
# --------------------------------------------------------------------------- #
FIFA_PLAYER_ATTRS = [
    "overall", "potential", "value_eur", "age",
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
]
N_STARTERS = 11

# Features de FORMA do time (temporais) — NÃO derivam de uma escalação estática;
# vêm do histórico (gold) e podem ser sobrescritas na simulação (*3*).
TEAM_FORM_FEATURES = ["win_rate_last10"]

_DEF = {"CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB"}
_MID = {"CDM", "LDM", "RDM", "CM", "LCM", "RCM", "LM", "RM", "CAM", "LAM", "RAM"}
_FWD = {"ST", "LS", "RS", "CF", "LF", "RF", "LW", "RW"}
_SECTOR_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
SECTOR_ORDER = ["GK", "DEF", "MID", "FWD"]

# Grupos de POSIÇÃO (mais finos que o setor) — capturam o "tipo" do jogador
# (zagueiro central x lateral, volante x meia-armador x ponta x centroavante).
# A FORMAÇÃO do time (ex.: 4-3-3) é a contagem por SETOR; a distribuição de
# POSIÇÕES é a contagem por GRUPO. Ambas viram features numéricas para o modelo.
_POSITION_GROUPS = {
    "gk": {"GK"},
    "cb": {"CB", "LCB", "RCB"},                 # zagueiros centrais
    "fb": {"LB", "RB", "LWB", "RWB"},           # laterais / alas
    "dm": {"CDM", "LDM", "RDM"},                # volantes
    "cm": {"CM", "LCM", "RCM"},                 # meias centrais
    "wm": {"LM", "RM"},                         # meias abertos
    "am": {"CAM", "LAM", "RAM"},                # meias-armadores
    "wf": {"LW", "RW", "LF", "RF"},             # pontas
    "st": {"ST", "LS", "RS", "CF"},             # centroavantes
}
POSITION_GROUP_ORDER = list(_POSITION_GROUPS)


def sector_of(position) -> str:
    """Posição → setor (GK/DEF/MID/FWD), igual a `macros.fifa.sector_of`."""
    p = (str(position) if position is not None else "").upper()
    if p == "GK":
        return "GK"
    if p in _DEF:
        return "DEF"
    if p in _MID:
        return "MID"
    if p in _FWD:
        return "FWD"
    return "MID"


def position_group_of(position) -> str:
    """Posição → grupo de posição (gk/cb/fb/dm/cm/wm/am/wf/st)."""
    p = (str(position) if position is not None else "").upper()
    for grp, members in _POSITION_GROUPS.items():
        if p in members:
            return grp
    return "cm"  # fallback: meia central (consistente com sector_of → MID)


def formation_features(positions: list, side: str) -> dict:
    """Features de FORMAÇÃO (contagem por setor) e POSIÇÃO (contagem por grupo).

    Recebe a lista de posições dos titulares de um lado e devolve
    `{side}_n_{gk,def,mid,fwd}` (formação) e `{side}_n_{grupo}` (distribuição de
    posições). Posições nulas são ignoradas. Tudo numérico, para entrar no modelo.
    """
    sectors = [sector_of(p) for p in positions if p is not None and not pd.isna(p)]
    groups = [position_group_of(p) for p in positions if p is not None and not pd.isna(p)]
    feats: dict = {}
    for sec in SECTOR_ORDER:
        feats[f"{side}_n_{sec.lower()}"] = float(sectors.count(sec))
    for grp in POSITION_GROUP_ORDER:
        feats[f"{side}_n_{grp}"] = float(groups.count(grp))
    return feats


def formation_string(positions: list) -> str:
    """Formação legível a partir das posições (ex.: "4-3-3"), excluindo o goleiro."""
    sectors = [sector_of(p) for p in positions if p is not None and not pd.isna(p)]
    parts = [sectors.count(s) for s in ("DEF", "MID", "FWD")]
    return "-".join(str(n) for n in parts if n) or "—"


def position_group_stat_features(players: list[dict], side: str) -> dict:
    """STATS MÉDIOS por GRUPO de posição: `{side}_grp_{grupo}_{attr}_mean`.

    Para cada grupo de posição (zagueiro central, lateral, volante, ponta…) tira a
    média de cada atributo FIFA dos jogadores daquele grupo no XI. Dá ao modelo a
    qualidade média POR posição (ex.: força dos pontas x dos volantes), e não só a
    média geral do time. Grupos vazios ficam NaN (caem no corte de 20% no treino).
    """
    by_grp: dict[str, list[dict]] = {g: [] for g in POSITION_GROUP_ORDER}
    for p in players:
        by_grp[position_group_of(p.get("position"))].append(p)
    feats: dict = {}
    for grp in POSITION_GROUP_ORDER:
        for a in FIFA_PLAYER_ATTRS:
            feats[f"{side}_grp_{grp}_{a}_mean"] = _safe_mean([p.get(a) for p in by_grp[grp]])
    return feats


def _safe_mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(vals)) if vals else np.nan


def order_lineup(players: list[dict]) -> list[dict]:
    """Ordena os jogadores por setor (GK→DEF→MID→FWD) e overall desc (como a gold)."""
    return sorted(
        players,
        key=lambda p: (_SECTOR_ORDER.get(sector_of(p.get("position")), 2),
                       -(p.get("overall") or 0)),
    )[:N_STARTERS]


def lineup_side_features(players: list[dict], side: str) -> dict:
    """Features de um lado a partir de uma escalação custom (mesmos nomes da gold).

    Produz `{side}_pNN_{name,position,<attrs>}`, agregados `{side}_xi_*` e os
    scores de time `{side}_team_{avg,attack,mid,def,gk}_score`. NÃO inclui forma
    (`win_rate_last10`), que é temporal e vem do histórico do time.
    """
    ordered = order_lineup(players)
    feats: dict = {}

    # slots por jogador
    for i, p in enumerate(ordered, start=1):
        prefix = f"{side}_p{i:02d}"
        feats[f"{prefix}_name"] = p.get("name")
        feats[f"{prefix}_position"] = p.get("position")
        for a in FIFA_PLAYER_ATTRS:
            feats[f"{prefix}_{a}"] = p.get(a)

    by_sector = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in ordered:
        by_sector[sector_of(p.get("position"))].append(p)

    feats[f"{side}_xi_count"] = len(ordered)
    for a in FIFA_PLAYER_ATTRS:
        feats[f"{side}_xi_{a}_mean"] = _safe_mean([p.get(a) for p in ordered])
    overalls = [p.get("overall") for p in ordered if p.get("overall") is not None]
    feats[f"{side}_xi_overall_max"] = float(np.max(overalls)) if overalls else np.nan
    feats[f"{side}_xi_overall_min"] = float(np.min(overalls)) if overalls else np.nan
    for sec in ("GK", "DEF", "MID", "FWD"):
        feats[f"{side}_xi_overall_mean_{sec}"] = _safe_mean(
            [p.get("overall") for p in by_sector[sec]]
        )

    # *1*/*2* — scores de time ponderados por posição (fórmulas iguais à gold).
    feats[f"{side}_team_avg_score"] = _safe_mean([p.get("overall") for p in ordered])
    feats[f"{side}_team_attack_score"] = _safe_mean(
        [((p.get("shooting") or 0) + (p.get("pace") or 0) + (p.get("dribbling") or 0)) / 3.0
         for p in by_sector["FWD"]]
    )
    feats[f"{side}_team_mid_score"] = _safe_mean(
        [((p.get("passing") or 0) + (p.get("dribbling") or 0)) / 2.0 for p in by_sector["MID"]]
    )
    feats[f"{side}_team_def_score"] = _safe_mean(
        [((p.get("defending") or 0) + (p.get("physic") or 0)) / 2.0 for p in by_sector["DEF"]]
    )
    feats[f"{side}_team_gk_score"] = _safe_mean([p.get("overall") for p in by_sector["GK"]])

    # *4* — FORMAÇÃO (contagem por setor) + distribuição de POSIÇÕES (por grupo) +
    # stats MÉDIOS por grupo de posição.
    feats.update(formation_features([p.get("position") for p in ordered], side))
    feats.update(position_group_stat_features(ordered, side))
    return feats


def team_form(store: "TeamFeatureStore", team: str) -> dict:
    """Forma (ex.: win_rate_last10) do time a partir do store, por nome resolvido."""
    key = _resolve_team_name(team, store)
    src = store.team_features[key] if key else store.fallback_features
    return {f: float(src.get(f, 0.0)) for f in TEAM_FORM_FEATURES}


def build_custom_match_row(
    home_players: list[dict],
    away_players: list[dict],
    feature_columns: list[str],
    home_form: dict | None = None,
    away_form: dict | None = None,
    fillna_value: float = 0.0,
) -> pd.DataFrame:
    """Linha de previsão a partir de DUAS escalações custom (usado pelo app).

    `{home,away}_form` injetam as features temporais (ex.: `win_rate_last10`); se
    None, ficam em 0. Colunas não-numéricas (nomes/posições) e ausentes recebem
    `fillna_value`, alinhando ao contrato do modelo (`feature_columns`).
    """
    values: dict = {}
    values.update(lineup_side_features(home_players, "home"))
    values.update(lineup_side_features(away_players, "away"))
    for f, v in (home_form or {}).items():
        values[f"home_{f}"] = v
    for f, v in (away_form or {}).items():
        values[f"away_{f}"] = v

    row = {}
    for col in feature_columns:
        v = values.get(col, fillna_value)
        if not isinstance(v, (int, float)) or pd.isna(v):
            v = fillna_value
        row[col] = v
    return pd.DataFrame([row], columns=feature_columns).astype("float32")


@dataclass
class TeamFeatureStore:
    team_features: dict[str, pd.Series]
    template_row: pd.Series
    feature_columns: list[str]
    team_name_aliases: dict[str, str]
    fallback_features: pd.Series


def add_position_features(gold_df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as features de FORMAÇÃO e POSIÇÃO à gold (devolve uma cópia).

    Lê as posições dos 11 slots de cada lado (`{side}_pNN_position`) e cria as
    contagens numéricas por SETOR (formação: `{side}_n_{gk,def,mid,fwd}`) e por
    GRUPO de posição (`{side}_n_{cb,fb,dm,cm,wm,am,wf,st}`). Idempotente: pode ser
    chamada mais de uma vez sem efeito colateral.
    """
    out = gold_df.copy()
    _add_position_features_inplace(out)
    return out


def _add_position_features_inplace(gold_df: pd.DataFrame) -> None:
    """Versão in-place de `add_position_features` (usada por `get_training_feature_columns`)."""
    for side in ("home", "away"):
        slots = [
            i for i in range(1, N_STARTERS + 1)
            if f"{side}_p{i:02d}_position" in gold_df.columns
        ]
        if not slots:
            continue
        positions = gold_df[[f"{side}_p{i:02d}_position" for i in slots]]
        sectors = positions.map(lambda v: sector_of(v) if pd.notna(v) else None)
        groups = positions.map(lambda v: position_group_of(v) if pd.notna(v) else None)
        # Contagens: FORMAÇÃO (por setor) e distribuição de POSIÇÕES (por grupo).
        for sec in SECTOR_ORDER:
            gold_df[f"{side}_n_{sec.lower()}"] = (sectors == sec).sum(axis=1).astype("float64")
        for grp in POSITION_GROUP_ORDER:
            gold_df[f"{side}_n_{grp}"] = (groups == grp).sum(axis=1).astype("float64")
        # Stats MÉDIOS por grupo de posição: média do atributo nos slots do grupo.
        grp_arr = groups.to_numpy()
        for attr in FIFA_PLAYER_ATTRS:
            attr_cols = [f"{side}_p{i:02d}_{attr}" for i in slots]
            if not all(c in gold_df.columns for c in attr_cols):
                continue
            vals = gold_df[attr_cols].to_numpy(dtype="float64")
            for grp in POSITION_GROUP_ORDER:
                masked = np.where(grp_arr == grp, vals, np.nan)
                with warnings.catch_warnings():  # silencia "Mean of empty slice"
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    gold_df[f"{side}_grp_{grp}_{attr}_mean"] = np.nanmean(masked, axis=1)


def get_training_feature_columns(gold_df: pd.DataFrame) -> list[str]:
    """Lista de features de treino — agora com FORMAÇÃO e POSIÇÃO do time.

    Enriquecimento (in place, idempotente): antes de selecionar as colunas, deriva
    das posições dos 11 slots (a) as contagens por SETOR (formação 4-3-3 etc.:
    `{side}_n_{gk,def,mid,fwd}`), (b) as contagens por GRUPO de posição
    (`{side}_n_{cb,fb,dm,cm,wm,am,wf,st}`) e (c) os STATS MÉDIOS por grupo de
    posição (`{side}_grp_{grupo}_{attr}_mean`), de modo que o modelo passe a "ver"
    a forma, a composição posicional e a qualidade média por posição do XI — não
    só os atributos médios do time inteiro.

    Critério de seleção: colunas numéricas com >=20% de não-nulos, excluindo alvos
    e contadores brutos (`do_not_use`). Como o `gold_df` é enriquecido in place, o
    `X = df[feature_columns]` do notebook já enxerga as novas colunas.
    """
    _add_position_features_inplace(gold_df)
    do_not_use = [
        "home_goals", "away_goals", "home_xi_count", "away_xi_count",
    ]
    thresh = len(gold_df) * 0.20
    return [
        col
        for col in gold_df.dropna(axis=1, thresh=thresh)
        .select_dtypes(include="number")
        .columns.tolist()
        if col not in do_not_use
    ]


def _to_team_perspective(
    match_row: pd.Series,
    side: str,
    allowed_side_columns: set[str],
) -> pd.Series:
    """Converte uma linha histórica em features do time (sem o prefixo de lado)."""
    prefix = f"{side}_"
    values = {}
    for col, value in match_row.items():
        if col.startswith(prefix) and col in allowed_side_columns:
            values[col[len(prefix) :]] = value
    return pd.Series(values, dtype="float64")


def build_team_feature_store(
    gold_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    team_name_aliases: dict[str, str] | None = None,
    match_type: str | None = "nation",
) -> TeamFeatureStore:
    """Monta o snapshot de features mais recente de cada SELEÇÃO (gold FIFA).

    Parâmetros
    ----------
    gold_df : DataFrame
        `data/gold_fifa_partidas` lido em pandas.
    match_type : str | None
        Filtra as partidas por tipo antes de montar (default "nation" → só
        seleções; use None para considerar tudo, inclusive clubes).
    """
    if match_type is not None and "match_type" in gold_df.columns:
        gold_df = gold_df[gold_df["match_type"] == match_type]

    # Ignora as linhas ESPELHADAS (:mirror) ao montar o snapshot por time — elas
    # são apenas augmentation de treino; as features do time já estão nas originais.
    if "match_id" in gold_df.columns:
        gold_df = gold_df[~gold_df["match_id"].astype(str).str.endswith(":mirror")]

    if feature_columns is None:
        feature_columns = get_training_feature_columns(gold_df)

    team_name_aliases = {**TEAM_NAME_ALIASES, **(team_name_aliases or {})}

    hist = gold_df.sort_values("match_date").copy()
    feature_column_set = set(feature_columns)

    side_agnostic_columns = sorted(
        {col[len("home_") :] for col in feature_columns if col.startswith("home_")}
        | {col[len("away_") :] for col in feature_columns if col.startswith("away_")}
    )
    template_row = pd.Series(0.0, index=side_agnostic_columns, dtype="float64")

    team_features: dict[str, pd.Series] = {}

    for team_col, side in [("home_team", "home"), ("away_team", "away")]:
        side_hist = hist.dropna(subset=[team_col])
        # Prefere o último jogo em que o time foi CASADO ao FIFA (features não-nulas).
        matched_col = f"{side}_matched"
        if matched_col in side_hist.columns:
            side_hist = side_hist[side_hist[matched_col] == True]  # noqa: E712
        latest_rows = side_hist.groupby(team_col, sort=False).tail(1)
        for _, row in latest_rows.iterrows():
            team = str(row[team_col])
            snapshot = template_row.copy()
            snapshot.update(_to_team_perspective(row, side, feature_column_set))
            # Não sobrescreve um snapshot já preenchido por um jogo MAIS recente do
            # outro lado: mantém o de maior match_date (hist está ordenado asc, e
            # processamos home depois away — então comparamos e ficamos com o último).
            team_features[team] = snapshot

    return TeamFeatureStore(
        team_features=team_features,
        template_row=template_row,
        feature_columns=feature_columns,
        team_name_aliases=team_name_aliases,
        fallback_features=pd.DataFrame(team_features).T.median(axis=0).astype("float64"),
    )


def _resolve_team_name(team: str, store: TeamFeatureStore) -> str | None:
    resolved = store.team_name_aliases.get(team, team)
    return resolved if resolved in store.team_features else None


def prepare_match_row(
    home_team: str,
    away_team: str,
    store: TeamFeatureStore,
    fillna_value: float = 0.0,
    use_fallback_for_missing: bool = True,
) -> pd.DataFrame:
    """Cria uma linha de previsão alinhada às colunas de treino (gold FIFA)."""
    home_key = _resolve_team_name(home_team, store)
    away_key = _resolve_team_name(away_team, store)
    if not use_fallback_for_missing and (home_key is None or away_key is None):
        missing = [team for team, key in [(home_team, home_key), (away_team, away_key)] if key is None]
        raise KeyError(f"No historical features found for: {missing}")

    home_features = store.team_features[home_key] if home_key else store.fallback_features
    away_features = store.team_features[away_key] if away_key else store.fallback_features

    row = {}
    for col in store.feature_columns:
        if col.startswith("home_"):
            row[col] = home_features.get(col[len("home_") :], fillna_value)
        elif col.startswith("away_"):
            row[col] = away_features.get(col[len("away_") :], fillna_value)
        else:
            row[col] = fillna_value

    match_df = pd.DataFrame([row], columns=store.feature_columns)
    return match_df.fillna(fillna_value)


def prepare_world_cup_matches(
    fixtures_df: pd.DataFrame,
    store: TeamFeatureStore,
    home_col: str = "HomeTeam",
    away_col: str = "AwayTeam",
    match_number_col: str = "MatchNumber",
    use_fallback_for_missing: bool = True,
) -> pd.DataFrame:
    """Cria uma linha pronta para o modelo por fixture da Copa (gold FIFA)."""
    rows = []
    metadata = []

    for _, fixture in fixtures_df.iterrows():
        home_team = fixture[home_col]
        away_team = fixture[away_col]

        # Pula placeholders de chaveamento (ex.: "1A", "2B", "3CDFGH").
        if any(str(team).startswith(("1", "2", "3")) for team in [home_team, away_team]):
            continue
        if any(team in {"To be announced", None} for team in [home_team, away_team]):
            continue

        match_row = prepare_match_row(
            home_team=home_team,
            away_team=away_team,
            store=store,
            use_fallback_for_missing=use_fallback_for_missing,
        )
        rows.append(match_row.iloc[0])
        metadata.append(
            {
                "MatchNumber": fixture.get(match_number_col),
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "RoundNumber": fixture.get("RoundNumber"),
                "Group": fixture.get("Group"),
                "DateUtc": fixture.get("DateUtc"),
                "home_team_resolved": store.team_name_aliases.get(home_team, home_team),
                "away_team_resolved": store.team_name_aliases.get(away_team, away_team),
                "home_team_has_history": _resolve_team_name(home_team, store) is not None,
                "away_team_has_history": _resolve_team_name(away_team, store) is not None,
            }
        )

    features_df = pd.DataFrame(rows).reset_index(drop=True)
    metadata_df = pd.DataFrame(metadata)
    return pd.concat([metadata_df, features_df], axis=1)


def build_lineup_feature_store(
    lineups_df: pd.DataFrame,
    feature_columns: list[str],
    hist_store: "TeamFeatureStore | None" = None,
    team_name_aliases: dict[str, str] | None = None,
) -> TeamFeatureStore:
    """Monta um TeamFeatureStore a partir de ESCALAÇÕES (ex.: o CSV da Copa),
    juntando com a FORMA do time (win rate) do `hist_store` (histórico/gold).

    Para cada seleção (`nationality`), as features de JOGADOR (slots p01..p11,
    agregados do XI, scores de time) vêm da escalação via `lineup_side_features`;
    as features TEMPORAIS (`TEAM_FORM_FEATURES`, ex.: win_rate_last10) vêm do
    `hist_store` por nome (com aliases). O store resultante entra direto em
    `prepare_match_row` / `simulate_world_cup`.

    `lineups_df` precisa de: `nationality`, `player_name`, `position` e os
    `FIFA_PLAYER_ATTRS` (o setor é recalculado de `position`).
    """
    side_cols = sorted(
        {c[len("home_"):] for c in feature_columns if c.startswith("home_")}
        | {c[len("away_"):] for c in feature_columns if c.startswith("away_")}
    )
    template = pd.Series(0.0, index=side_cols, dtype="float64")

    team_features: dict[str, pd.Series] = {}
    for team, grp in lineups_df.groupby("nationality"):
        players = [
            {"name": r["player_name"], "position": r["position"],
             **{a: r.get(a) for a in FIFA_PLAYER_ATTRS}}
            for _, r in grp.iterrows()
        ]
        feats = lineup_side_features(players, "home")  # prefixo home_
        snap = template.copy()
        for k, v in feats.items():
            agn = k[len("home_"):]
            if agn in template.index and isinstance(v, (int, float)) and not pd.isna(v):
                snap[agn] = v
        # forma do time (win_rate_last10 etc.) do histórico
        if hist_store is not None:
            for f, v in team_form(hist_store, str(team)).items():
                if f in template.index:
                    snap[f] = v
        team_features[str(team)] = snap

    aliases = {**TEAM_NAME_ALIASES, **(team_name_aliases or {})}
    fallback = (pd.DataFrame(team_features).T.median(axis=0).astype("float64")
                if team_features else template)
    return TeamFeatureStore(
        team_features=team_features, template_row=template,
        feature_columns=list(feature_columns), team_name_aliases=aliases,
        fallback_features=fallback,
    )
