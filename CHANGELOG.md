# CHANGELOG

## 2026-06-15 — App: poder MUDAR a formação
- Antes só dava p/ trocar jogador dentro do mesmo setor; a FORMA do XI ficava
  travada no XI padrão. Adicionado **seletor de Formação** por lado:
  `Padrão` (XI exato do time) + `4-3-3, 4-4-2, 4-5-1, 4-2-3-1, 3-4-3, 3-5-2,
  5-3-2, 5-4-1`. Mudar a formação redefine as posições/setores dos 11 slots
  (preenchidos com os melhores do setor), mudando as features de setor do modelo.
- Slots agora têm chave por (seleção, formação) → trocar formação/time dá selects
  novos no XI daquela combinação; as trocas do usuário persistem.
- **Validação (AppTest)**: trocar Brasil p/ 5-3-2 muda a previsão
  (22.4/27.0/50.6 → 21.9/38.2/39.9); trocar o goleiro também muda
  (22.4 → 17.9% de vitória). Título do campo mostra a formação.

## 2026-06-15 — bolt_pipeliner 0.2.6 → 0.2.7
- Atualizada a cópia vendorizada `_boltpipeliner/bolt_pipeliner` para **0.2.7**
  (instalada via `pip install bolt_pipeliner==0.2.7` e copiada). Só 5 arquivos
  mudaram: `__init__.py` (versão), `generators/documentation.py` e 3 templates de
  doc. **O runtime de ETL (runner, bases, config, selection, sessions) é idêntico
  ao 0.2.6 → zero impacto no pipeline.**
- Novidades do 0.2.7 (geração de docs): coluna **`parent`** (linhagem de colunas
  entre tabelas) no schema e **auto-geração de `outputs/schema/schema.csv`**
  inferindo o schema direto dos parquet de saída.
- Regerados os artefatos com `python generate.py documentation`:
  `outputs/documentation/*.html`, `outputs/schema/schema.py` (agora lista TODAS as
  tabelas atuais, incl. FIFA, automaticamente) e o novo `outputs/schema/schema.csv`
  (1492 linhas, 23 tabelas, 225 colunas com linhagem `parent`).
- **Validação**: `bolt_pipeliner.__version__ == 0.2.7`; `load_config` + bases +
  `macros.bases` OK; `gold_fifa_partidas` re-roda sem erro; geração de docs exit 0.

## 2026-06-15 — App: jogadores criados ficam SÓ no navegador (deploy público)
- "Criar jogador" não escreve mais em `app/app_data/custom_players.csv` (que seria
  COMPARTILHADO entre todos os visitantes e falha em FS efêmero/somente-leitura).
  Agora os jogadores criados vivem em `st.session_state` (por usuário/sessão) —
  isolados entre visitantes e sem escrita no servidor.
- Aba ➕ ganhou **⬇️ Baixar meus jogadores (CSV)** e **importar CSV**, p/ o usuário
  persistir/recarregar a própria lista localmente (a sessão se perde ao recarregar).
- **Validação (AppTest)**: criar jogador NÃO gera arquivo no servidor; fica em
  `session_state` e selecionável na mesma sessão; sessão nova NÃO vê os jogadores
  de outra (isolamento por usuário); simular segue OK.

## 2026-06-14 — Notebook: simular a Copa com o XI custom (CSV)
- Novo `model_notebooks/world_cup_simulation_custom_xi.ipynb`: simula a Copa 2026
  usando os XIs atualizados de `world_cup_2026_starting_xi_custom_players.csv`
  (48 seleções) + a forma (win rate) do histórico, reusando os 3 modelos
  LightGBM (resultado + gols casa/fora) e `simulate_world_cup`.
- Novo helper `world_cup_features_fifa.build_lineup_feature_store(...)`: monta um
  `TeamFeatureStore` a partir de escalações (features de jogador do CSV) juntando
  `win_rate_last10` do `hist_store`. Entra direto em `prepare_match_row`/`simulate_world_cup`.
- **Validação**: notebook executado (10/10 células, 0 erros); store custom com 48
  seleções (Brasil xi_overall_mean 83.9 do CSV + win_rate 0.4 do histórico);
  Brasil×Alemanha placar provável 1–1; campeão da simulação: **France**; o bracket
  inclui o placar (home_goals/away_goals) de cada jogo via os modelos de gols.

## 2026-06-14 — Fix: seleção "em branco" ao trocar de time (Streamlit)
- **Bug**: ao trocar a seleção, os slots ficavam em branco. Causa: o reset
  apagava as chaves `{side}_p*` do `session_state` a cada troca (padrão frágil que
  podia deixar os selectboxes vazios no navegador).
- **Fix**: chaves de widget agora são POR SELEÇÃO (`{side}_{team}_p{n}`). Ao trocar
  de time, os selectboxes são novos e já iniciam no XI padrão daquele time — sem
  apagar `session_state`. As escolhas do usuário persistem por seleção. Removida
  toda a lógica de reset.
- Campo de futebol agora renderiza numa sub-coluna de metade da largura
  (estável em qualquer versão do Streamlit) em vez de `width="content"`.
- **Validação (AppTest)**: troca por TODAS as 199 seleções sem exceção; XIs padrão
  corretos (Brasil→Alisson Becker, Alemanha→Neuer); simular e criar jogador OK.

## 2026-06-14 — App: XI da Copa (CSV) + win rate + campo menor
- O app Streamlit agora carrega `model_notebooks/world_cup_2026_starting_xi_custom_players.csv`
  (48 seleções × 11, stats de jogador ATUALIZADOS) como XI titular padrão e pool
  selecionável dessas seleções. `sector` é recalculado de `position` (o CSV vem em
  PT). Dedup por (seleção, jogador) com prioridade **criados > CSV > FIFA base**
  (stats do CSV vencem).
- **Join com a seleção**: o win rate (e demais features de forma) continua vindo
  do histórico via `team_form(store, ...)` — as 48 seleções do CSV resolvem para
  o store pelos aliases existentes (validado: 48/48, ex.: USA→United States,
  IR Iran→Iran). A linha de previsão combina stats do CSV (jogadores) + win rate
  do histórico (seleção).
- Seletor de seleção prioriza as 48 da Copa.
- **Campo de futebol 50% menor** (figsize 4.2×6.0 → 2.1×3.0, `width="content"`).
- **Validação (AppTest)**: Brasil×França 23,6/23,1/53,3% (0×3); USA×IR Iran
  54,7/24,4/20,9% (1×0) com XIs do CSV e win rate via alias; `home_p01_overall`
  usa o stat do CSV (Alisson 89); win rate Brasil/USA = 0,4 do store.

## 2026-06-13 — Modelos de gols tunados por Optuna
- Os 2 modelos de placar (gols casa/fora) agora são **tunados por Optuna**
  (antes usavam params fixos), minimizando **neg log-loss** em CV 5-fold
  (calibração de probabilidade — métrica certa p/ distribuição de placar).
  Mesma busca de hiperparâmetros do modelo de resultado, controlada por
  `WC_OPTUNA_TRIALS` (default 30). Atualizados: célula do notebook
  `tune_goal_model` + `train_goal_models.py`.
- Modelos re-treinados e salvos (Optuna, 6 trials neste ambiente — use
  `WC_OPTUNA_TRIALS=30` p/ a busca completa):
  - `fifa_best_model.lgb` (resultado, re-tunado via notebook),
  - `fifa_home_goals_model.lgb` — best neg_log_loss ≈ −1.4284 (1932 árvores),
  - `fifa_away_goals_model.lgb` — best neg_log_loss ≈ −1.4285 (3000 árvores).
- **Validação**: os 3 modelos carregam e preveem; Brasil×Alemanha → placar
  provável 1–2; app (AppTest) OK com os modelos atualizados (Brasil×França
  19,4/19,2/61,4%).
- Nota: a busca Optuna de 6 classes é pesada; a re-execução headless completa do
  notebook é lenta neste sandbox, então os 2 modelos de gols foram treinados pelo
  `train_goal_models.py` (lógica IDÊNTICA à célula do notebook). O código do
  notebook está atualizado (Optuna p/ os 3 modelos, `N_TRIALS` env default 30) —
  rode "Run All" no seu ambiente p/ atualizar as saídas e o tuning completo.


## 2026-06-13 — Ciclo 2 (modelos de placar + app interativo)

### Removido (worldcupapi.com)
- A API exigia assinatura paga (key retornava 401 "no data access"). Removidos:
  `model_notebooks/fetch_worldcup_lineups.py`, o CSV de escalações, e as funções
  de escalação externa (`norm_name`, `load_external_lineups`, `build_pool_index`,
  `pick_starting_xi`, `resolve_lineup_players`) de `world_cup_features_fifa.py`,
  além do override em `prepare_app_data.py`. `default_lineups`/`players_pool`
  voltaram ao XI padrão do FIFA (validado: 17.364 jogadores, 193 seleções).

### Modelos de placar (2 novos LightGBM) — DONE
- `train_goal_models.py` + células no notebook treinam classificadores de GOLS do
  mandante e do visitante, classes **0,1,2,3,4,5+** (gols ≥5 agrupados), mesmas
  252 features do modelo de resultado. Salvos: `fifa_home_goals_model.lgb` /
  `fifa_away_goals_model.lgb` (+ metas com `feature_columns`, `classes`, `labels`).
- **Validação**: Brasil×Alemanha → placar mais provável **1–2**; distribuições
  somam 1 e batem com o `predict_goal_distributions`.

### world_cup_simulation.py — DONE
- `_model_proba` (aceita Booster ou LGBMClassifier), `predict_goal_distributions`,
  `predict_match_full`; `MatchPrediction` ganhou `home_goals`/`away_goals`;
  `sample_match`/`simulate_*` aceitam `goal_models` e registram o placar previsto.

### Notebook — DONE
- Adicionadas células dos 2 modelos de gols (treino + save + exemplo de
  distribuição) após a montagem do `store`.
- Célula do optuna agora usa `N_TRIALS = int(os.environ.get("WC_OPTUNA_TRIALS","30"))`
  (default 30; defina a env var p/ rodar mais rápido).
- Pipeline RE-EXECUTADO de ponta a ponta (kernel `.venv` `analise-venv`): 19/20
  células, 0 erros; os 3 modelos foram re-salvos (resultado + gols casa/fora).
  ⚠️ A execução headless aqui usou poucos trials de optuna por tempo — re-rode o
  notebook com `WC_OPTUNA_TRIALS=30` (default) p/ o tuning completo do modelo de
  resultado.

### App Streamlit (reescrito) — DONE
- Abas **⚽ Simular** e **➕ Criar jogador**.
- Simular mostra os 3 modelos: resultado (W/D/L) + gols mandante (0..5+) + gols
  visitante (0..5+) + placar mais provável.
- Jogadores escaláveis = SÓ os da seleção escolhida (`nationality`), por setor.
- Criar jogador → salva em `app/app_data/custom_players.csv` e passa a aparecer
  na seleção (com atributos definidos pelo usuário).
- **Validação (AppTest headless)**: carrega (2 abas, 26 selectboxes); simular →
  Brasil×França 17,9/22,7/59,4% e placar "Brasil 1 × 2 França"; troca p/ Argentina
  → 171/171 opções argentinas (filtro nacional OK); criar "Teste Craque" (Brasil)
  → persistido e selecionável após reload.

### Dependências
- Instalado no `.venv`: `optuna`, `nbclient`, `ipykernel` (p/ rodar o notebook);
  kernel `analise-venv` registrado.


Mudanças e validações deste ciclo (features extras do modelo FIFA + app Streamlit).
Datas no formato ISO. Validações rodadas com `../.venv/bin/python`.

## [em andamento] 2026-06-12

### Planejado
- gold_fifa_partidas: espelhamento (59.970 linhas) + scores de time + win streak.
- Ajustes em world_cup_features_fifa.py / world_cup_simulation.py.
- App Streamlit + modelo dummy.

### gold_fifa_partidas (DONE)
- **Espelhamento HOME↔AWAY**: `_mirror()` troca todas as colunas `home_*`↔`away_*`
  e inverte `result` (H↔A) e `target` (W↔L); `match_id` ganha sufixo `:mirror`.
  Tabela passou de **29.985 → 59.970 linhas** (329 colunas).
- **Scores de time** (`_side_aggregate_exprs`): `{lado}_team_avg_score` (overall
  médio do XI), `{lado}_team_attack_score` (FWD: shooting/pace/dribbling),
  `{lado}_team_mid_score` (MID: passing/dribbling), `{lado}_team_def_score`
  (DEF: defending/physic), `{lado}_team_gk_score` (GK: overall).
- **Win streak** (`_win_rate_features`): `{lado}_win_rate_last10` = vitórias nas
  últimas 10 partidas / 10 (shift(1) → sem vazamento).
- **Validação**: shape (59970, 329); mirror confere troca de times, inversão de
  target e swap de scores/win-rate; `win_rate_last10 ∈ [0,1]`; alvos ficaram
  simétricos (W=L=21.709, D=16.552) — efeito esperado do espelhamento.

### world_cup_features_fifa.py (DONE)
- Novos helpers para o app: `sector_of`, `order_lineup`, `lineup_side_features`,
  `team_form`, `build_custom_match_row` (monta a linha de previsão a partir de
  DUAS escalações custom, com as MESMAS fórmulas da gold) + constantes
  `FIFA_PLAYER_ATTRS`, `TEAM_FORM_FEATURES=['win_rate_last10']`.
- `build_team_feature_store` agora ignora linhas `:mirror` (determinístico).
- **Validação**: `build_custom_match_row(XI Brasil, XI Alemanha)` reproduz
  EXATAMENTE `prepare_match_row('Brazil','Germany')` (0.263/0.457/0.280) —
  prova que a feature do app == feature da gold.

### world_cup_simulation.py (DONE)
- `predict_match_proba`/`sample_match`/`simulate_*` aceitam
  `team_feature_overrides` (e `home/away_overrides`) p/ alterar características de
  feature por time na simulação (*3*), ex.: `{"Brazil": {"win_rate_last10": 1.0}}`.
- **Validação**: override de `win_rate_last10`→1.0 elevou P(vitória) Brasil×Irã
  de 0.615 → 0.646.

### Modelo dummy (DONE)
- `train_dummy_fifa_model.py`: treina LightGBM (sem optuna) sobre o conjunto de
  features atual e salva `fifa_best_model.lgb` (Booster) + `fifa_best_model_meta.json`
  (`feature_columns`, `classes`). **252 features**, classes `['D','L','W']`.
  ⚠️ Modelo DUMMY p/ teste — re-treinar via `world_cup_lightgbm_simulation_fifa.ipynb`.

### App Streamlit (DONE)
- `app/prepare_app_data.py` → `app/app_data/{players_pool,default_lineups}.parquet`
  (pool de 17.364 jogadores do snapshot mais recente; XI padrão de 193 seleções,
  escolhendo o snapshot mais completo).
- `app/streamlit_app.py`: tela HOME/AWAY, campo desenhado (matplotlib), seletor de
  seleção, 11 jogadores por setor trocáveis por busca, botão Simular → previsão
  W/D/L via `fifa_best_model.lgb`.
- **Validação (headless AppTest)**: app sobe (HTTP 200), 24 selectboxes, sem
  exceção no load nem no Simular; Brasil×França = 19,5/17,6/62,9%;
  trocar away→Argentina = 9,5/25,4/65,1% (consistente com a predição direta).

### Escalações externas (lineups melhores) — DONE (parcial)
- **Bati na API ao vivo** (worldcupapi.com): descobri pela coleção Postman oficial
  o base real `https://api.worldcupapi.com`, os endpoints `/squads?key=&team_id=`
  e `/lineups?key=&match_id=`, e os `team_id` de 42 seleções (embutidos no fetcher).
  Sem key válida a API responde **HTTP 401** `{"success":false,"error":"This API
  key and secret do not have access..."}` — confirmado batendo em
  `/squads?team_id=1448`. Ou seja: só falta a SUA key (registrar em /register).
- Entregue mesmo assim:
  - `model_notebooks/fetch_worldcup_lineups.py` — fetcher (defensivo) que o usuário
    roda com a SUA key (`WORLDCUPAPI_KEY`), gerando `worldcup_lineups.csv`
    (colunas `team,player_name,position`). Tem `--dump` p/ inspecionar o JSON e
    `--teams-from`/`--base-url` p/ ajustar.
  - `world_cup_features_fifa.py`: `load_external_lineups`, `build_pool_index`,
    `resolve_lineup_players`, `pick_starting_xi`, `norm_name` — usa a escalação
    externa quando o jogador EXISTE no pool do FIFA; **jogador ausente do
    dataframe é PULADO e o slot vai para o jogador PADRÃO** (XI do FIFA). Como
    `/squads` traz o elenco (~26), `pick_starting_xi` escolhe um XI plausível
    (melhor GK + melhores por overall) dos jogadores presentes no pool.
  - `app/prepare_app_data.py`: aplica o CSV (se existir) ao montar
    `default_lineups.parquet` (coluna `source` = 'external'/'default'); sem CSV,
    mantém o XI padrão do FIFA.
- **Validação** (CSVs sintéticos):
  (a) 8 reais + 1 fake p/ Brasil, 6 p/ "USA"→United States: fake PULADO, Brasil = 8
      externos + 3 default, alias de time aplicado.
  (b) "elenco" de 28 linhas (4 GKs + 2 fakes): `pick_starting_xi` devolveu 11
      jogadores com EXATAMENTE 1 GK, fakes pulados.
  Fetcher testado ao vivo (key inválida → 401 tratado). App segue passando no
  AppTest. CSVs sintéticos removidos (estado entregue = XI padrão FIFA).

### Dependências / como rodar
- Instalado no `.venv`: `lightgbm 4.6.0`, `streamlit 1.52`, `matplotlib`.
- Rodar o app:
  ```bash
  cd football_analysis
  ../.venv/bin/python -m app.prepare_app_data          # 1x (gera os dados do app)
  ../.venv/bin/python model_notebooks/train_dummy_fifa_model.py   # 1x (modelo dummy)
  ../.venv/bin/streamlit run app/streamlit_app.py
  ```
