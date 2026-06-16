"""Diamond — tabela de partidas com features normalizadas por fonte (rank/quantil).

Por que esta camada existe
--------------------------
As features de forma (médias e p75 de 18m) vêm de provedores diferentes
(StatsBomb e FBref/Opta) que NÃO estão na mesma escala — em especial o xG, que
usa modelos distintos. Empilhar as linhas numa tabela de treino mistura duas
distribuições. Aqui alinhamos as features por **rank/quantil dentro de cada
fonte**.

Decisões (conforme acordado)
----------------------------
• Método = **rank/quantil**: cada valor vira seu percentil em (0,1] dentro do
  grupo. É robusto a outliers e a escalas diferentes entre provedores.
• Grupo = **apenas `source`** (NÃO por competição). Isso é proposital: queremos
  CORRIGIR a diferença de provedor, mas PRESERVAR a diferença de força entre
  ligas — um time de uma liga fraca deve mesmo ter feature menor que o de uma
  liga forte da mesma fonte. Normalizar por competição apagaria esse sinal.
• As colunas BRUTAS são mantidas; adicionamos versões com sufixo `_q`.
• Os `target_*` NÃO são normalizados (são o que se quer prever).

Resultado: para cada feature `X` (ex.: home_team_form_xg_total_mean_18m) surge
`X_q` em (0,1], comparável entre fontes. Linhas sem histórico (feature nula)
permanecem nulas no `_q`.
"""
import polars as pl


def _feature_columns(columns: list[str]) -> list[str]:
    """Colunas de feature a normalizar: médias e percentis de 18m.

    Exclui chaves, contadores de suporte (`n_partidas`, `n_titulares`, ...) e
    os alvos (`target_*`).
    """
    return [
        c
        for c in columns
        if (c.endswith("_mean_18m") or c.endswith("_p75_18m"))
        and not c.startswith("target_")
    ]


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    gold = input_tables["gold"]

    feature_cols = _feature_columns(gold.columns)

    # Rank/quantil DENTRO de cada fonte: rank médio / nº de valores não-nulos do
    # grupo → percentil em (0,1]. `.over("source")` particiona por provedor;
    # valores nulos permanecem nulos (rank não os pontua).
    norm_exprs = [
        (
            pl.col(c).rank(method="average") / pl.col(c).count()
        ).over("source").alias(f"{c}_q")
        for c in feature_cols
    ]

    return gold.with_columns(norm_exprs)
