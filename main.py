#!/usr/bin/env python
"""Executa o pipeline (engine: polars).

Driver leve, sem dependência de `typer` nem `pyspark`. O runner padrão do
bolt_pipeliner cria uma SparkSession por padrão, mas a engine deste projeto é
100% polars — então passamos um "sentinela" de spark (que o ETLBaseParquetPolars
ignora) para pular essa criação e evitar exigir pyspark.

Exemplos:
    python main.py                 # todas as camadas, na ordem de dependência
    python main.py --bronze        # só a bronze
    python main.py --silver --gold # silver e depois gold
    python main.py --select gold_partidas_features   # um job específico
    python main.py -v              # verboso
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_VENDOR = _HERE / "_boltpipeliner"
if _VENDOR.is_dir():
    sys.path.insert(0, str(_VENDOR))
# Garante que `etl.*` (jobs e helpers) seja importável independente do CWD.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bolt_pipeliner.runner import run as runner_run  # noqa: E402

_LAYER_FLAGS = ("flatfile", "bronze", "silver", "gold", "diamond")


class _NoSpark:
    """Sentinela não-None passado ao runner para pular a criação da SparkSession.

    O ETLBaseParquetPolars recebe `spark` apenas via **kwargs e nunca o utiliza,
    portanto este objeto vazio é suficiente para uma execução puramente polars.
    """


def main(argv: list[str]) -> None:
    config = "configs/etl_config.yaml"
    verbose = False
    select = None
    layers: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-v", "--verbose"):
            verbose = True
        elif arg in ("-c", "--config"):
            i += 1
            config = argv[i]
        elif arg in ("-s", "--select"):
            i += 1
            select = argv[i]
        elif arg.startswith("--") and arg[2:] in _LAYER_FLAGS:
            layers.append(arg[2:])
        else:
            raise SystemExit(f"Argumento não reconhecido: {arg!r}")
        i += 1

    runner_run(
        config,
        layers=layers or None,
        select=select,
        verbose=verbose,
        spark=_NoSpark(),  # evita create_session() / pyspark
    )


if __name__ == "__main__":
    main(sys.argv[1:])
