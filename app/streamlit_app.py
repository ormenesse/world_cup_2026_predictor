#!/usr/bin/env python3
"""App Streamlit — simulador de partida de seleções (3 modelos FIFA LightGBM).

Abas:
  • ⚽ Simular — duas seleções (HOME/AWAY), campo desenhado, e SÓ jogadores da
    seleção escolhida (pelo `nationality` do FIFA) podem ser escalados/trocados.
    Ao simular, mostra 3 modelos: resultado (W/D/L) + distribuição de GOLS do
    mandante (0..5+) + distribuição de GOLS do visitante (0..5+) + placar provável.
  • ➕ Criar jogador — cria um jogador novo numa seleção (caso não exista no FIFA);
    fica salvo em `app/app_data/custom_players.csv` e passa a aparecer na seleção.

Como rodar:
    cd football_analysis
    ../.venv/bin/python -m app.prepare_app_data        # 1x (pool + XIs padrão)
    python model_notebooks/train_goal_models.py        # 1x (modelos de gols)
    ../.venv/bin/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Circle, Rectangle

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_NB = _ROOT / "model_notebooks"
for _p in (_ROOT, _NB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from world_cup_features_fifa import (  # noqa: E402
    FIFA_PLAYER_ATTRS,
    build_custom_match_row,
    build_team_feature_store,
    formation_string,
    sector_of,
    team_form,
)

_APP_DATA = _HERE / "app_data"
_GOLD = _ROOT / "data" / "gold_fifa_partidas" / "part-0.parquet"
# Jogadores criados ficam SÓ no navegador do usuário (session_state), nunca no
# servidor — assim cada visitante tem os seus e nada é compartilhado (ok p/ deploy).
_CUSTOM_KEY = "custom_players_df"
# XI titular ATUALIZADO da Copa 2026 (stats de jogador mais recentes), por seleção.
_WC_XI_CSV = _NB / "world_cup_2026_starting_xi_custom_players.csv"

_MODELS = {
    "result": _NB / "fifa_best_model.lgb",
    "home_goals": _NB / "fifa_home_goals_model.lgb",
    "away_goals": _NB / "fifa_away_goals_model.lgb",
}

_SECTOR_LINES = {"GK": 8.0, "DEF": 30.0, "MID": 55.0, "FWD": 82.0}
_SECTOR_ORDER = ["GK", "DEF", "MID", "FWD"]
# Posições oferecidas ao criar jogador (cobrem todos os setores).
_POSITIONS = ["GK", "CB", "LB", "RB", "LWB", "RWB", "CDM", "CM", "CAM",
              "LM", "RM", "LW", "RW", "CF", "ST"]

# Formações selecionáveis (nome -> 11 posições). Mudar a formação muda os SETORES
# do XI (e, portanto, as features do modelo). "Padrão" usa o XI exato do time.
_DEFAULT_FORMATION = "Padrão"
_FORMATIONS = {
    "4-3-3": ["GK", "RB", "CB", "CB", "LB", "CM", "CM", "CM", "RW", "ST", "LW"],
    "4-4-2": ["GK", "RB", "CB", "CB", "LB", "RM", "CM", "CM", "LM", "ST", "ST"],
    "4-5-1": ["GK", "RB", "CB", "CB", "LB", "RM", "CM", "CM", "CM", "LM", "ST"],
    "4-2-3-1": ["GK", "RB", "CB", "CB", "LB", "CDM", "CDM", "CAM", "RW", "LW", "ST"],
    "3-4-3": ["GK", "CB", "CB", "CB", "RM", "CM", "CM", "LM", "RW", "ST", "LW"],
    "3-5-2": ["GK", "CB", "CB", "CB", "RM", "CM", "CM", "CM", "LM", "ST", "ST"],
    "5-3-2": ["GK", "RB", "CB", "CB", "CB", "LB", "CM", "CM", "CM", "ST", "ST"],
    "5-4-1": ["GK", "RB", "CB", "CB", "CB", "LB", "RM", "CM", "CM", "LM", "ST"],
}


# --------------------------------------------------------------------------- #
# Carregamento
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_base_pool() -> pd.DataFrame:
    return pd.read_parquet(_APP_DATA / "players_pool.parquet")


@st.cache_data(show_spinner=False)
def load_default_lineups() -> pd.DataFrame:
    return pd.read_parquet(_APP_DATA / "default_lineups.parquet")


_POOL_COLS = ["player_name", "position", "sector", "nationality", *FIFA_PLAYER_ATTRS]


def load_custom_players() -> pd.DataFrame:
    """Jogadores criados pelo usuário — SÓ no navegador dele (session_state).

    Não há leitura/escrita em disco do servidor: cada sessão tem os seus, nada é
    compartilhado entre usuários (seguro p/ deploy público, ex.: Vercel).
    """
    df = st.session_state.get(_CUSTOM_KEY)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_POOL_COLS)
    for c in _POOL_COLS:
        if c not in df.columns:
            df[c] = None
    return df[_POOL_COLS]


def save_custom_players(df: pd.DataFrame) -> None:
    """Guarda os jogadores criados no session_state (por usuário/navegador)."""
    st.session_state[_CUSTOM_KEY] = df[_POOL_COLS].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_wc_xi() -> pd.DataFrame:
    """XI titular ATUALIZADO da Copa (CSV) — stats de jogador mais recentes.

    Recalcula `sector` (GK/DEF/MID/FWD) a partir de `position` (o CSV traz setor
    em português), p/ ficar consistente com o campo, os agregados e o modelo.
    """
    if not _WC_XI_CSV.exists():
        return pd.DataFrame(columns=_POOL_COLS)
    df = pd.read_csv(_WC_XI_CSV)
    df["sector"] = df["position"].map(sector_of)
    for c in _POOL_COLS:
        if c not in df.columns:
            df[c] = None
    return df[_POOL_COLS]


def get_pool() -> pd.DataFrame:
    """Pool de jogadores = FIFA base + XI atualizado da Copa (CSV) + criados.

    Prioridade na deduplicação por (seleção, jogador): criados > CSV da Copa >
    FIFA base — assim os stats ATUALIZADOS do CSV vencem os do FIFA base.
    """
    frames = [f for f in (load_custom_players(), load_wc_xi(), load_base_pool()) if len(f)]
    pool = pd.concat(frames, ignore_index=True)
    return pool.drop_duplicates(subset=["nationality", "player_name"], keep="first")


@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for name, path in _MODELS.items():
        booster = lgb.Booster(model_file=str(path))
        meta = json.loads(path.with_name(path.stem + "_meta.json").read_text(encoding="utf-8"))
        out[name] = (booster, meta)
    feature_columns = out["result"][1]["feature_columns"]
    return out, feature_columns


@st.cache_resource(show_spinner=True)
def load_store(feature_columns: tuple[str, ...]):
    return build_team_feature_store(pd.read_parquet(_GOLD), list(feature_columns))


def _player_label(name: str, idx: dict) -> str:
    row = idx.get(name)
    if row is None:
        return str(name)
    return f"{name} — {int(row.get('overall') or 0)} ({row.get('position') or '?'})"


# --------------------------------------------------------------------------- #
# Campo de futebol
# --------------------------------------------------------------------------- #
def draw_pitch(lineup: list[dict], idx: dict, title: str):
    fig, ax = plt.subplots(figsize=(2.1, 3.0))  # 50% menor que o original (4.2 x 6.0)
    ax.add_patch(Rectangle((0, 0), 100, 100, color="#2e7d32"))
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color="white", lw=2)
    ax.plot([0, 100], [50, 50], color="white", lw=1.5)
    ax.add_patch(Circle((50, 50), 9, fill=False, color="white", lw=1.5))
    for y0 in (0, 82):
        ax.add_patch(Rectangle((25, y0), 50, 18, fill=False, color="white", lw=1.2))

    by_sector: dict[str, list[dict]] = {s: [] for s in _SECTOR_ORDER}
    for slot in lineup:
        by_sector.get(slot["sector"], by_sector["MID"]).append(slot)
    for sec, players in by_sector.items():
        if not players:
            continue
        y = _SECTOR_LINES[sec]
        xs = [(i + 1) * 100 / (len(players) + 1) for i in range(len(players))]
        for x, slot in zip(xs, players):
            name = slot["player_name"]
            ovr = (idx.get(name) or {}).get("overall")
            ax.add_patch(Circle((x, y), 4.2, color="white", ec="#1b5e20", lw=1.5, zorder=3))
            if ovr is not None and pd.notna(ovr):
                ax.text(x, y, str(int(ovr)), ha="center", va="center",
                        fontsize=7, fontweight="bold", color="#1b5e20", zorder=4)
            short = str(name).split()[-1][:11] if name else "—"
            ax.text(x, y - 6.5, short, ha="center", va="center",
                    fontsize=6.5, color="white", zorder=4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")
    return fig


# --------------------------------------------------------------------------- #
# Escalação por seleção (SÓ jogadores da nationality escolhida)
# --------------------------------------------------------------------------- #
def default_lineup(team: str, lineups: pd.DataFrame, wc_xi: pd.DataFrame) -> list[dict]:
    """XI padrão da seleção: usa o CSV ATUALIZADO da Copa quando existe; senão,
    o XI padrão do FIFA (`default_lineups.parquet`). Ordena por setor
    (GK→DEF→MID→FWD) e overall desc, atribuindo `slot` 1..11."""
    wc = wc_xi[wc_xi["nationality"] == team]
    if not wc.empty:
        order = {s: i for i, s in enumerate(_SECTOR_ORDER)}
        wc = wc.assign(_so=wc["sector"].map(order)).sort_values(
            ["_so", "overall"], ascending=[True, False])
        return [
            {"slot": i, "position": r["position"], "sector": r["sector"],
             "player_name": r["player_name"]}
            for i, (_, r) in enumerate(wc.iterrows(), start=1)
        ]
    rows = lineups[lineups["team_name"] == team].sort_values("slot")
    return [
        {"slot": int(r["slot"]), "position": r["position"],
         "sector": r["sector"] or sector_of(r["position"]), "player_name": r["player_name"]}
        for _, r in rows.iterrows()
    ]


def render_side(side: str, teams: list[str], lineups: pd.DataFrame,
                wc_xi: pd.DataFrame, pool: pd.DataFrame, idx: dict) -> tuple[str, list[dict]]:
    label = "🏠 HOME" if side == "home" else "✈️ AWAY"
    st.subheader(label)
    default_team = "Brazil" if side == "home" else "France"
    team = st.selectbox("Seleção", teams,
                        index=teams.index(default_team) if default_team in teams else 0,
                        key=f"{side}_team")

    # Pool SÓ da seleção (nationality == team), por setor.
    team_pool = pool[pool["nationality"] == team]
    by_sector = {
        sec: team_pool[team_pool["sector"] == sec].sort_values("overall", ascending=False)
        for sec in _SECTOR_ORDER
    }

    base = default_lineup(team, lineups, wc_xi)

    # Seletor de FORMAÇÃO. "Padrão" = XI exato do time; as demais reorganizam os
    # setores (muda DEF/MID/FWD e, portanto, as features de setor do modelo).
    formation = st.selectbox(
        "Formação", [_DEFAULT_FORMATION, *_FORMATIONS],
        index=0, key=f"{side}_{team}_formation",
    )
    # Slots (posição, setor) da formação escolhida. No "Padrão", usa o XI do time;
    # nas demais, define a posição de cada slot pela formação.
    if formation == _DEFAULT_FORMATION:
        slot_specs = [(s["position"], s["sector"], s["player_name"]) for s in base]
    else:
        slot_specs = [(p, sector_of(p), None) for p in _FORMATIONS[formation]]

    # Chaves de widget por (seleção, formação): trocar time/formação dá selects
    # novos no XI padrão daquela combinação; as trocas do usuário persistem.
    tkey = f"{side}_{team}_{formation}"
    sector_rank = {sec: 0 for sec in _SECTOR_ORDER}

    lineup: list[dict] = []
    with st.expander("Escalação — trocar jogadores (só desta seleção)", expanded=False):
        if team_pool.empty:
            st.info("Sem jogadores desta seleção no pool do FIFA. Crie jogadores na aba ➕.")
        for i, (pos, sec, fixed_name) in enumerate(slot_specs, start=1):
            names_sec = list(by_sector[sec]["player_name"])
            # default do slot: XI exato (Padrão) ou o j-ésimo melhor do setor.
            if fixed_name is not None:
                default_name = fixed_name
            else:
                j = sector_rank[sec]
                default_name = (names_sec[j] if j < len(names_sec)
                                else (names_sec[-1] if names_sec else None))
            sector_rank[sec] += 1

            options = names_sec.copy()
            if default_name and default_name not in options:
                options = [default_name] + options
            key = f"{tkey}_p{i}"
            if not options:
                chosen = None
            elif key in st.session_state and st.session_state[key] in options:
                chosen = st.selectbox(pos, options,
                                      format_func=lambda n: _player_label(n, idx), key=key)
            else:
                idx0 = options.index(default_name) if default_name in options else 0
                chosen = st.selectbox(pos, options, index=idx0,
                                      format_func=lambda n: _player_label(n, idx), key=key)
            lineup.append({"slot": i, "position": pos, "sector": sec, "player_name": chosen})

    enriched = []
    for slot in lineup:
        if not slot["player_name"]:
            continue
        attrs = idx.get(slot["player_name"], {})
        enriched.append({
            "name": slot["player_name"], "position": slot["position"],
            **{a: attrs.get(a) for a in FIFA_PLAYER_ATTRS},
        })

    title = formation if formation != _DEFAULT_FORMATION else formation_string(
        [s["position"] for s in lineup])
    # Campo ~50% menor: renderiza numa sub-coluna de metade da largura.
    pitch_col, _ = st.columns(2)
    with pitch_col:
        st.pyplot(draw_pitch(lineup, idx, f"{team}  ·  {title}"), width="stretch")
    return team, enriched


# --------------------------------------------------------------------------- #
# Abas
# --------------------------------------------------------------------------- #
def tab_simulate(teams, lineups, wc_xi, pool, idx, models, feature_columns, store):
    col_home, col_away = st.columns(2)
    with col_home:
        home_team, home_players = render_side("home", teams, lineups, wc_xi, pool, idx)
    with col_away:
        away_team, away_players = render_side("away", teams, lineups, wc_xi, pool, idx)

    st.divider()
    if not st.button("▶️ Simular partida", type="primary", width="stretch"):
        return

    row = build_custom_match_row(
        home_players, away_players, feature_columns,
        home_form=team_form(store, home_team), away_form=team_form(store, away_team),
    )
    vals = row[feature_columns].values

    res_booster, res_meta = models["result"]
    res = dict(zip(res_meta["classes"], res_booster.predict(vals)[0]))
    p_home, p_draw, p_away = res.get("W", 0.0), res.get("D", 0.0), res.get("L", 0.0)

    hg_booster, hg_meta = models["home_goals"]
    ag_booster, ag_meta = models["away_goals"]
    hg = hg_booster.predict(vals)[0]
    ag = ag_booster.predict(vals)[0]
    hg_labels, ag_labels = hg_meta["labels"], ag_meta["labels"]
    h_ml = hg_labels[int(hg.argmax())]
    a_ml = ag_labels[int(ag.argmax())]

    st.subheader("Resultado")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Vitória {home_team}", f"{p_home:.1%}")
    c2.metric("Empate", f"{p_draw:.1%}")
    c3.metric(f"Vitória {away_team}", f"{p_away:.1%}")
    st.success(f"Placar mais provável: **{home_team} {h_ml} × {a_ml} {away_team}**")

    g1, g2 = st.columns(2)
    with g1:
        st.caption(f"Gols do {home_team} (mandante)")
        st.bar_chart(pd.DataFrame({"prob": list(hg)}, index=hg_labels))
    with g2:
        st.caption(f"Gols do {away_team} (visitante)")
        st.bar_chart(pd.DataFrame({"prob": list(ag)}, index=ag_labels))


def tab_create_player(teams):
    st.subheader("Criar jogador para uma seleção")
    st.caption("Útil quando o jogador não existe no FIFA. Fica salvo **só no seu navegador** "
               "(nesta sessão) e aparece na seleção — não é compartilhado com outras pessoas.")

    # Importar jogadores salvos antes (CSV baixado pelo próprio usuário).
    up = st.file_uploader("Importar jogadores criados (CSV baixado antes)", type="csv",
                          key="import_custom")
    if up is not None and st.session_state.get("_imported_name") != up.name:
        try:
            imported = pd.read_csv(up)
            for col in _POOL_COLS:
                if col not in imported.columns:
                    imported[col] = None
            merged = pd.concat([load_custom_players(), imported[_POOL_COLS]], ignore_index=True)
            merged = merged.drop_duplicates(subset=["nationality", "player_name"], keep="last")
            save_custom_players(merged)
            st.session_state["_imported_name"] = up.name
            st.success(f"Importados {len(imported)} jogador(es) do CSV.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Não consegui ler o CSV: {exc}")

    with st.form("create_player", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        nationality = c1.selectbox("Seleção (nationality)", teams)
        name = c2.text_input("Nome do jogador")
        position = c3.selectbox("Posição", _POSITIONS)
        a, b, c, d = st.columns(4)
        overall = a.slider("Overall", 40, 99, 75)
        potential = b.slider("Potential", 40, 99, 80)
        age = c.slider("Idade", 16, 42, 24)
        value_eur = d.number_input("Valor (€)", 0, 300_000_000, 5_000_000, step=500_000)
        e, f, g, h, i, j = st.columns(6)
        pace = e.slider("Pace", 1, 99, 70)
        shooting = f.slider("Shooting", 1, 99, 70)
        passing = g.slider("Passing", 1, 99, 70)
        dribbling = h.slider("Dribbling", 1, 99, 70)
        defending = i.slider("Defending", 1, 99, 70)
        physic = j.slider("Physic", 1, 99, 70)
        submitted = st.form_submit_button("➕ Adicionar jogador", type="primary")

    if submitted:
        if not name.strip():
            st.error("Informe o nome do jogador.")
            return
        rec = {
            "player_name": name.strip(), "position": position,
            "sector": sector_of(position), "nationality": nationality,
            "overall": overall, "potential": potential, "value_eur": value_eur, "age": age,
            "pace": pace, "shooting": shooting, "passing": passing,
            "dribbling": dribbling, "defending": defending, "physic": physic,
        }
        df = pd.concat([load_custom_players(), pd.DataFrame([rec])], ignore_index=True)
        save_custom_players(df)   # só na sessão do navegador (sem escrever no servidor)
        st.success(f"Jogador **{name}** ({position}, {overall}) adicionado a **{nationality}**. "
                   "Selecione-o na aba ⚽ Simular.")

    custom = load_custom_players()
    if len(custom):
        st.markdown("##### Jogadores criados (nesta sessão)")
        st.dataframe(custom[["player_name", "nationality", "position", "overall"]],
                     width="stretch", hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Baixar meus jogadores (CSV)", custom.to_csv(index=False),
            file_name="meus_jogadores.csv", mime="text/csv", width="stretch",
            help="Salve no seu computador p/ reimportar depois (a sessão se perde ao recarregar).",
        )
        if d2.button("🗑️ Apagar todos os criados", width="stretch"):
            save_custom_players(pd.DataFrame(columns=_POOL_COLS))
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Simulador FIFA — Seleções", layout="wide")
    st.title("⚽ Simulador de Partida — Seleções (modelos FIFA)")
    st.caption("Resultado (W/D/L) + gols do mandante e do visitante (0..5+), via LightGBM. "
               "O modelo considera a formação (ex.: 4-3-3), as posições e a qualidade "
               "média por posição do XI.")

    lineups = load_default_lineups()
    wc_xi = load_wc_xi()
    pool = get_pool()
    idx = pool.drop_duplicates("player_name").set_index("player_name").to_dict("index")
    models, feature_columns = load_models()
    store = load_store(tuple(feature_columns))
    # seleções: as 48 da Copa (CSV) primeiro; + XI FIFA + nationalities de criados.
    wc_teams = sorted(wc_xi["nationality"].dropna().unique())
    others = sorted((set(lineups["team_name"]) | set(pool["nationality"].dropna())) - set(wc_teams))
    teams = wc_teams + others

    if len(wc_xi):
        st.caption(f"XI da Copa 2026 carregado do CSV ({len(wc_teams)} seleções, stats "
                   "atualizados) + win rate da seleção do histórico.")

    tab_sim, tab_new = st.tabs(["⚽ Simular", "➕ Criar jogador"])
    with tab_sim:
        tab_simulate(teams, lineups, wc_xi, pool, idx, models, feature_columns, store)
    with tab_new:
        tab_create_player(teams)


if __name__ == "__main__":
    main()
