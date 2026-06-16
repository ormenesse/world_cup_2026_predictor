#!/usr/bin/env python3
"""Treina 2 modelos LightGBM de PLACAR (gols do mandante e do visitante) com OPTUNA.

Cada modelo é um classificador multiclasse com 6 classes: 0, 1, 2, 3, 4, 5+
(gols >= 5 agrupados). Os hiperparâmetros são otimizados por Optuna minimizando o
log-loss em CV 5-fold (calibração de probabilidade — adequado p/ distribuição de
placar), igual ao espírito do tuning do modelo de resultado. Usa o MESMO conjunto
de features (`get_training_feature_columns`).

NÃO mexe no modelo de resultado (`fifa_best_model.lgb`).

Salva (em model_notebooks/):
  • fifa_home_goals_model.lgb + fifa_home_goals_model_meta.json
  • fifa_away_goals_model.lgb + fifa_away_goals_model_meta.json

Nº de trials do Optuna: env `WC_OPTUNA_TRIALS` (default 30).
Uso:  cd model_notebooks && python train_goal_models.py
      WC_OPTUNA_TRIALS=8 python train_goal_models.py   # mais rápido
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from world_cup_features_fifa import get_training_feature_columns

_HERE = Path(__file__).resolve().parent
_GOLD = _HERE.parent / "data" / "gold_fifa_partidas" / "part-0.parquet"
GOAL_CAP = 5  # gols >= 5 viram a classe "5+"
N_TRIALS = int(os.environ.get("WC_OPTUNA_TRIALS", "30"))


def tune_goal_model(X: pd.DataFrame, y_goals: pd.Series, n_classes: int):
    """Otimiza (Optuna, neg log-loss) e treina o classificador final de gols."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss",
            "boosting_type": "gbdt", "verbosity": -1, "random_state": 42, "n_jobs": 1,
            "n_estimators": trial.suggest_int("n_estimators", 10, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        return cross_val_score(model, X, y_goals, cv=cv, scoring="neg_log_loss", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    best = lgb.LGBMClassifier(
        objective="multiclass", num_class=n_classes, metric="multi_logloss",
        boosting_type="gbdt", verbosity=-1, random_state=42, n_jobs=1, **study.best_params,
    )
    best.fit(X, y_goals)
    return best, study.best_value


def main() -> None:
    df = pd.read_parquet(_GOLD)
    feature_columns = get_training_feature_columns(df)
    X = df[feature_columns].fillna(0).astype("float32")
    print(f"[goals] features: {len(feature_columns)} | linhas: {len(df)} | trials: {N_TRIALS}")

    for side, target in (("home", "home_goals"), ("away", "away_goals")):
        y_goals = df[target].clip(upper=GOAL_CAP).astype(int)
        classes = sorted(y_goals.unique().tolist())
        model, best_score = tune_goal_model(X, y_goals, len(classes))
        model_path = _HERE / f"fifa_{side}_goals_model.lgb"
        meta_path = _HERE / f"fifa_{side}_goals_model_meta.json"
        model._Booster.save_model(str(model_path))
        labels = [str(c) if c < GOAL_CAP else f"{GOAL_CAP}+" for c in classes]
        meta_path.write_text(json.dumps({
            "feature_columns": feature_columns, "classes": classes, "labels": labels,
            "target": target, "goal_cap": GOAL_CAP,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[goals] {target}: best neg_log_loss={best_score:.4f} | classes {labels} -> {model_path.name}")


if __name__ == "__main__":
    main()
