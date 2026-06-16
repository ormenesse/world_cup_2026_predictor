# Pipeline de dados — análise de futebol (engine: **polars**)

Transforma os CSVs brutos de `dados/` em **uma única tabela final orientada a
partidas** (`data/gold_partidas_features/`), com:

- **Features**: médias móveis de **18 meses** (parametrizável) das estatísticas de
  cada **time** e dos **titulares** escalados, calculadas **estritamente antes**
  do mês de cada jogo (sem data leakage).
- **Targets** (prefixo `target_`): o que aconteceu de fato no jogo — resultado,
  gols por lado, chutes/chutes no alvo, faltas, cartões amarelos, expulsões e
  vermelhos por lado.

**TUDO fica na GOLD** (`gold_partidas_features`): features brutas + targets +
versões `_q` normalizadas por fonte. A camada **diamond está vazia** por opção.

Resultado atual (só StatsBomb): **3.776 partidas × 692 colunas** — médias 18m
(geral, por time, por setor de posição), percentis p75, índices defensivos /
on-off, 15 `target_*`, e as 320 versões `_q` normalizadas por fonte.

---

## Como obter os dados (coletor unificado)

Os dados são baixados por **um único script** direto em `data/raw/`
(substitui o antigo `main.py` da raiz):

```bash
cd football_analysis
pip install statsbombpy soccerdata pandas       # ou ../.venv/bin/pip install ...

# baixa StatsBomb (open data) + FBref (temporada) + FBref (ligas extras por partida)
python -m etl.sources.fetch_fbref

# atalhos úteis:
python -m etl.sources.fetch_fbref --max-competicoes 2     # teste rápido (StatsBomb)
python -m etl.sources.fetch_fbref --skip-fbref-temporada --skip-fbref-partidas  # só StatsBomb
python -m etl.sources.fetch_fbref --only-extras "BRA-Serie A"  # só uma liga extra
```

Saída: `data/raw/sb_*.csv`, `data/raw/fbref_*.csv` e
`data/raw/fbref/{schedule,player_match_summary,lineup}/`.
StatsBomb é grátis/sem rate-limit; o FBref faz scraping (rate-limit ~1 req/3s —
comece com `--max-competicoes`/`--only-extras` e use o cache em `~/soccerdata/`).

## Como rodar o pipeline

> Requisitos: `polars`, `pyarrow`, `pyyaml` (NÃO precisa de pyspark — a engine é polars).

```bash
# da pasta football_analysis/
python main.py                 # todas as camadas, na ordem de dependência
python main.py --flatfile      # só uma camada
python main.py --silver --gold
python main.py -s gold_partidas_features   # um job específico (com upstream via +)
python main.py -v              # verboso
```

Lê os dados de `data/raw/` e grava os datasets Parquet de cada camada em `data/`.

---

## Arquitetura (medallion)

```
fetch_fbref (coletor) ──► data/raw/*.csv  (+ data/raw/fbref/…)
                   │  (flatfile)
                   ▼
   flatfile_sb_*          flatfile_fbref_*        ← CSV bruto → Parquet
   (StatsBomb, pass-through)  (FBref, resolve cabeçalho de 3 linhas)
                   │  (bronze)
                   ▼
   bronze_partidas      bronze_escalacoes        bronze_stats_jogador
   (datas, placar,      (escalação COMPLETA:     (stats por jogador/partida
    mando, year_month)   titular + cartões)       + calendário + lado + id)
                   │  (silver — agregados mensais parametrizáveis)
                   ▼
   silver_jogador_year_month     silver_time_year_month
   (jogador × year_month)        (time × year_month, totais por partida)
                   │  (gold)
                   ▼
            gold_partidas_features   ← TABELA FINAL (1 linha por partida)
```

### Camadas

| Camada | Job | O que faz |
|---|---|---|
| flatfile | `ff_passthrough` | materializa cada CSV do StatsBomb como Parquet |
| flatfile | `ff_fbref` | idem para o FBref, resolvendo o cabeçalho hierárquico de 3 linhas |
| bronze | `bronze_partidas` | tipa datas/placares, deriva `year_month`/`month_index` |
| bronze | `bronze_escalacoes` | escalação **completa** + `is_starter` + contagem de cartões/expulsões |
| bronze | `bronze_stats_jogador` | stats por jogador/partida + data, lado (home/away) e `player_id` |
| silver | `silver_jogador_year_month` | estatísticas por **jogador × mês** (todo o elenco) |
| silver | `silver_time_year_month` | estatísticas por **time × mês** (totais por partida) |
| gold | `gold_partidas_features` | junta tudo: features de forma 18m + targets |

> **Por que duas silvers?** Conforme pedido, há um agregado por **jogador×mês** e
> outro por **time×mês**. A gold usa o de time para a "forma do time" e o de
> jogador para a "forma média dos titulares". A silver de jogador cobre o
> **elenco inteiro** — hoje a gold filtra só titulares (`is_starter`), mas trocar
> para o elenco todo é remover esse filtro, **sem reprocessar a silver**.

---

## Parametrização — `configs/feature_config.yaml`

Todo o "negócio" mora aqui (nenhum código precisa mudar):

```yaml
rolling_window_months: 18      # janela das médias móveis
player_match_stats:            # quais colunas de sb_stats_jogador entram
  - gols
  - chutes_no_alvo
  - xg_total
  - ...
aggregations:                  # agregações por estatística nas silvers
  - mean
  - sum
percentiles:                   # percentis calculados na gold (0.75 = p75)
  - 0.75
percentile_stats:              # variáveis importantes para os percentis
  - gols
  - chutes_no_alvo
  - xg_total
  - ...
```

- Adicione/remova estatísticas em `player_match_stats` → silvers e gold se
  adaptam automaticamente (os nomes das colunas seguem o padrão
  `{stat}_{agg}` e `{lado}_{escopo}_{stat}_mean_18m`).
- `{stat}_sum` e `n_partidas` são **sempre** gravados nas silvers, pois a média
  móvel de 18m da gold é **ponderada** (`Σsoma / Σpartidas`) — mais correta que
  média de médias mensais.

---

## Esquema da tabela final (`gold_partidas_features`)

**Identificação**: `match_id, match_date, year_month, month_index,
competition_name, season, home_team, away_team, home_score, away_score`.

**Features (18 meses, calculadas só com o passado)**:
- `home_team_form_<stat>_mean_18m` / `away_team_form_<stat>_mean_18m`
  — forma recente do **time** (de `silver_time_year_month`).
- `home_lineup_<stat>_mean_18m` / `away_lineup_<stat>_mean_18m`
  — média, entre os **titulares**, da forma de 18m de cada jogador
  (de `silver_jogador_year_month`).
- `*_team_form_n_partidas_18m`, `*_lineup_n_titulares`, `*_lineup_n_com_historico`
  — contadores de suporte (confiabilidade da média).
- `home_team_form_<stat>_p75_18m` / `home_lineup_<stat>_p75_18m` (e `away_`)
  — **percentil 75** das variáveis importantes (`percentile_stats` no YAML).
  Capta o "teto"/consistência recente, complementando a média. Calculado sobre
  a distribuição **por partida** na janela (não a partir das silvers), pois
  percentil não é reconstrutível de soma/contagem.

**Targets (`target_`)**:
- `target_result` (H/D/A), `target_home_goals`, `target_away_goals`
- `target_<lado>_shots_on_target`, `target_<lado>_shots`, `target_<lado>_fouls`
- `target_<lado>_yellow_cards`, `target_<lado>_expulsions`, `target_<lado>_red_cards`

### Sem data leakage
A janela é `[mês_do_jogo − 18, mês_do_jogo − 1]` (estritamente antes). Jogos sem
histórico na janela ficam com features **nulas** (ex.: a partida mais antiga de
2009, ou as seleções da Women's Euro 2025, cujos times não têm jogos prévios no
dataset — embora as jogadoras tenham histórico de clube, captado pelas features
de `lineup`).

---

## Fontes de dados e como adicionar mais (multi-fonte)

O pipeline é **multi-fonte**. Hoje:

- **StatsBomb** (`data/raw/sb_*.csv`) — fonte primária, sempre presente.
- **FBref** (via `soccerdata`) — fonte **opcional** para trazer competições que
  o StatsBomb não cobre (2ª divisões EU, Brasileirão A/B, Argentina, Libertadores).

As camadas silver/gold são **agnósticas de fonte**: elas leem as tabelas de
UNIÃO (`bronze_all_*`), que empilham todas as fontes no mesmo schema canônico.

### Como puxar o FBref

Faz parte do **coletor unificado** (ver "Como obter os dados"):
`python -m etl.sources.fetch_fbref` baixa StatsBomb + FBref(temporada) +
FBref(ligas extras por partida). Para só as ligas extras de uma competição:
`python -m etl.sources.fetch_fbref --skip-statsbomb --skip-fbref-temporada --only-extras "BRA-Serie A"`.
Edite as competições/temporadas em `configs/fbref_sources.yaml`. O coletor injeta
as ligas no `league_dict` do `soccerdata` (2ª divisões e América do Sul não vêm
por padrão) e **pula** competições sem estatística por jogador no FBref.

> **Limitações do FBref** (`stat_type='summary'`): traz gols, Sh, SoT, xG, passes
> (Cmp/Att), interceptações, blocks, take-ons e cartões — mas **não** tem faltas,
> key passes, pressões, duelos, recuperações (ficam nulas). Não há `player_id` no
> FBref → sintetizamos `fbref:<nome>` (não cruze ids entre fontes sem crosswalk).

### Fluxo multi-fonte na bronze

```
StatsBomb:  flatfile_sb_* ─► bronze_partidas / bronze_escalacoes / bronze_stats_jogador
FBref:      data/raw/fbref/* ─► bronze_fbref_partidas / _escalacoes / _stats_jogador
                                   │  (macros/fbref.py → schema canônico)
                                   ▼
              bronze_union ─► bronze_all_partidas / _escalacoes / _stats_jogador
                                   │
                                   ▼  (silver + gold leem os bronze_all_*)
```

Jobs FBref e união usam a base **`ETLTolerantPolars`**: se o FBref não foi
baixado, as entradas viram vazias e a união fica idêntica a só-StatsBomb
(**zero regressão** — validado: 3.776 partidas inalteradas).

### Adicionar OUTRA fonte (ex.: API-Football)
1. Escreva um fetcher em `etl/sources/` que grave parquet bruto em `data/raw/<fonte>/`.
2. Crie um adapter `etl/_<fonte>_adapter.py` que produza o schema canônico
   (mesmas colunas de `bronze_stats_jogador` etc.) — espelhe `_fbref_adapter.py`.
3. Adicione 3 jobs `bronze_<fonte>_*` (base tolerante) e inclua-os nos
   `input_tables` dos 3 `bronze_union`. Silver/gold não mudam.

## Avaliação por posição e qualidade defensiva (zagueiro/meio-campo)

xG mede ataque; para avaliar **defesa e setores** o pipeline classifica cada
titular em `position_group` (GK/DEF/MID/FWD, derivado das posições do
StatsBomb/FBref na bronze) e gera, no gold:

- **Ataque/forma por setor**: `home_DEF_setor_<stat>_mean_18m`,
  `home_MID_setor_<stat>_mean_18m`, ... (média 18m dos titulares de cada setor).
- **Índices defensivos individuais** (parametrizáveis em `feature_config`):
  `def_acoes` (interceptações+bloqueios+cortes+recuperações+duelos) e `def_erros`
  (erros+driblado+gol contra+faltas) → fluem como qualquer stat; o gold deriva
  `*_def_confiabilidade_18m = ações/(ações+erros)`.
- **Solidez defensiva coletiva (on-off)** — o sinal mais robusto para "quão bom é
  o zagueiro NAQUELE time":
  - `*_team_def_{gols_sofridos,xg_sofrido}_mean_18m` = baseline do time
    (silver_time_defensivo);
  - `*_lineup_def_*` e `*_<SETOR>_setor_def_*` = sofrido pelo time COM os
    titulares / com o setor em campo (silver_jogador_defensivo);
  - `*_onoff_{gols_sofridos,xg_sofrido}_mean_18m` = (com o jogador/setor) −
    (baseline do time). **NEGATIVO ⇒ time sofre menos ⇒ bom defensivamente.**

> `xg_sofrido` = soma do xG do time **adversário**; `gols_sofridos` = gols do
> adversário (placar). Validação de sanidade: interceptações/jogo por setor
> DEF (1,26) > MID (0,92) > FWD (0,41), como esperado.

Ajuste `position_groups` no `feature_config.yaml` para incluir/remover setores
(controla também a largura da tabela).

## Unificação de IDs (chaves consistentes entre fontes)

Para que tudo seja realmente unificado numa tabela só:

- **`player_id` — canônico por NOME, idêntico entre fontes** (`player_uid_expr`):
  nome normalizado (NFKD → sem acento → minúsculas → só `[a-z0-9 ]`). Assim o
  MESMO jogador recebe o MESMO id no StatsBomb e no FBref (o StatsBomb usa id
  inteiro próprio e o FBref não tem id — o nome é a única chave comum).
  Ex.: "Mário Figueira Fernandes" → `mario figueira fernandes`.
- **`match_id` — com namespace de fonte** (feito no `bronze_union`):
  `statsbomb:<id>` / `fbref:<game_id>`. Partidas são disjuntas entre fontes, então
  o namespace garante unicidade e deixa a origem explícita (sem colisão).

> **Trade-off honesto:** como o `player_id` é unificado por nome, um jogador que
> apareça nas DUAS fontes terá seu histórico de 18m combinado — misturando
> escalas de provedores diferentes (ver abaixo). Na prática o overlap é pequeno
> (competições disjuntas) e a normalização `_q` por fonte (no nível da partida)
> mitiga o efeito. Nomes grafados de forma diferente entre provedores não casam
> (limitação sem uma crosswalk oficial de ids).

## Comparabilidade entre fontes e normalização (IMPORTANTE)

StatsBomb e FBref (Opta/StatsPerform) são **provedores diferentes**: definições
de evento divergem e, sobretudo, o **xG usa modelos distintos** — os valores
**não estão na mesma escala**. Como as duas fontes cobrem competições
**disjuntas** (sem jogos em comum), não há como calibrar uma contra a outra nos
mesmos jogos.

Salvaguardas no design:
- Toda linha carrega a coluna **`source`** (`statsbomb`/`fbref`).
- As features de uma partida usam **apenas histórico da mesma fonte** (o
  `player_id` do FBref é `"fbref:<nome>"`, e os times de ligas disjuntas não se
  cruzam) — logo **não há mistura de provedores dentro de uma mesma média**.

Para tornar as features comparáveis ao **empilhar as linhas num modelo**, a
**própria gold** gera, para cada feature de 18m, uma versão **rank/quantil em
(0,1] calculada DENTRO de cada `source`** (sufixo `_q`), mantendo as colunas
brutas ao lado:

- **rank/quantil** → robusto a outliers e a escalas de provedores diferentes;
- **agrupado só por `source`** (não por competição) **de propósito**: corrige a
  incompatibilidade de provedor, mas **preserva a diferença de força entre
  ligas** (um time de liga fraca deve mesmo pontuar abaixo de um de liga forte
  da mesma fonte). Os `target_*` não são normalizados.

> Tudo isso fica na GOLD (colunas brutas + `*_q`); **não há camada diamond**.

## Notas de implementação

- **Bucket único `data/`**: o framework grava cada saída como diretório
  `data/<layer>_<nome>/`, mas a leitura entre camadas anexa `.parquet`. Por isso
  as entradas downstream usam glob `"<layer>_<nome>/*.parquet"`.
- **Base flatfile robusta** (`macros.bases.ETLFlatfilePolars`): força inferência de
  schema sobre o arquivo inteiro, evitando quebras em colunas com valores mistos
  (ex.: co-técnicos em `sb_partidas`).
- **Sem pyspark**: `main.py` chama o runner com um sentinela de spark, já que a
  engine é polars.
- **Lógica reutilizável vive em `macros/`** (qualquer função usada por VÁRIAS
  tabelas): `macros/features.py` (config, calendário, posição→setor, parsing de
  cartões/titularidade, agregações e médias/percentis móveis ponderados),
  `macros/fbref.py` (leitor de CSV de 3 cabeçalhos + adapter FBref→canônico),
  `macros/bases.py` (bases de ETL: flatfile robusta, tolerante e leitura em
  streaming do CSV gigante do FIFA) e `macros/fifa.py` (normalização/fuzzy de nome
  de time, seleção do XI, snapshot as-of, config FIFA). Os jobs de cada camada
  ficam curtos e importam de `macros.*`. (Antes em `etl/_features.py` /
  `etl/_fbref_adapter.py` / `etl/_bases.py`, removidos nesta reestruturação.)

## Pipeline FIFA (dataset partida × ratings de jogador)

Além da tabela de futebol (StatsBomb/FBref), o mesmo medallion produz
`data/gold_fifa_partidas/` — **1 linha por partida** cruzando os RESULTADOS
(`data/raw/fifa_matches/schedule.parquet`) com os ratings de jogador do FIFA
(`data/data_fifa/fifa_aggregated.csv`, ~7 GB). Substitui o antigo script
standalone `etl/build_fifa_player_match_dataset.py` (removido).

```
flatfile_fifa_players  (scan_csv streaming + projeção das ~20 colunas usadas)
flatfile_fifa_schedule (passthrough do parquet)
        │ (bronze)
        ▼
bronze_fifa_players  (tipa snapshot/atributos; chaves de time normalizadas)
bronze_fifa_matches  (result/target W-D-L; chaves home/away normalizadas)
        │ (silver)
        ▼
silver_fifa_team_snapshot (XI titular de cada time em cada snapshot — long)
silver_fifa_match_snapshot (cada partida + snapshot do FIFA vigente — as-of join)
        │ (gold)
        ▼
gold_fifa_partidas  ← 11 titulares de cada lado (`{lado}_p01..p11_*`) + score
                       agregado do time (médias/máx/mín do XI e por setor) + alvos
```

- **Sem vazamento temporal**: cada partida usa o snapshot do FIFA mais recente
  com data <= data do jogo (`join_asof` backward).
- **Casamento de nome** (schedule ↔ FIFA): normalização (sem acento/pontuação/
  stopwords + sufixo de UF do Brasil) → exato → fuzzy (difflib) → contenção de
  tokens, com aliases de clube/seleção (`macros.fifa`). Clubes casam dentro do
  pool de `fifa_league_ids` da competição (`configs/fifa_match_sources.yaml`),
  resolvendo a colisão Serie A Itália=31 vs Brasil=7.
- **Cobertura** (ambos os lados casados): 77,6% das 29.985 partidas (italy_a
  90,9%, world_cup 100%; brazil_a menor por nomes divergentes).
- **CSV de 7 GB**: lido em streaming com projeção de colunas pela base
  `macros.bases.ETLFifaPlayersFlatfilePolars` (não carrega o arquivo inteiro).
