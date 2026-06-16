"""Bronze (união) — concatena a MESMA tabela canônica vinda de várias fontes.

Módulo genérico reutilizado por 3 jobs (partidas, escalações, stats por
jogador): cada job declara em `input_tables` as versões por fonte
(ex.: `bronze_partidas` do StatsBomb + `bronze_fbref_partidas` do FBref) e este
process_data as empilha.

Detalhes:
  • Fontes vazias (ex.: FBref ainda não baixado) são ignoradas.
  • Chaves de identificação (`match_id`, `player_id`, `*_id`) são uniformizadas
    para texto, pois o StatsBomb usa ids inteiros e o FBref usa ids textuais.
    Isso evita conflito de tipo no concat e mantém os joins do gold consistentes.
  • `how="diagonal_relaxed"` alinha colunas por nome (preenchendo ausentes com
    nulo) e harmoniza tipos divergentes para um supertipo comum.

Se houver só uma fonte (StatsBomb), o resultado é idêntico ao de antes —
garantindo zero regressão quando o FBref não está presente.
"""
import polars as pl


def _normalize_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Uniformiza colunas de id para Utf8 (ints do StatsBomb vs textos do FBref)."""
    id_cols = [
        c for c in df.columns
        if c == "match_id" or c == "player_id" or c.endswith("_id")
    ]
    if not id_cols:
        return df
    return df.with_columns([pl.col(c).cast(pl.Utf8) for c in id_cols])


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    frames = [
        _normalize_ids(df)
        for df in input_tables.values()
        if df is not None and not df.is_empty()
    ]
    if not frames:
        return pl.DataFrame()
    out = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")

    # NAMESPACE do match_id por fonte → esquema unificado e SEM colisão entre
    # provedores (partidas são disjuntas). Ex.: "statsbomb:3754239",
    # "fbref:a1b2c3". Feito por linha via a coluna `source`. O player_id NÃO é
    # prefixado de propósito: é canônico por nome, igual entre fontes (unificado).
    if "match_id" in out.columns and "source" in out.columns:
        out = out.with_columns(
            (pl.col("source") + pl.lit(":") + pl.col("match_id").cast(pl.Utf8)).alias("match_id")
        )
    return out
