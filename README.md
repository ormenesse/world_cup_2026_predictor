# football_analysis

Bolt Pipeliner project — engine: **polars**, layers: **flatfile, bronze, silver, gold, diamond**.

This project ships with a vendored copy of `bolt_pipeliner` under `_boltpipeliner/` so the shims work even without a pip install.

## Usage

```bash
# Run the pipeline (all layers, or pass --bronze / --silver / ...)
python main.py

# Regenerate downstream artifacts (DAGs, docs, layer scripts, ...)
python generate.py documentation
python generate.py all

# Or use the full CLI directly
python bolt.py run --bronze
python bolt.py test
```

## ML

- `models/` — production training/loading code (`train_example.py`)
- `model_notebooks/` — experimentation surface for iterating on a model
- Suggested home for ML ETL jobs: the **diamond** layer (downstream of gold).
- **MLflow is recommended** for experiment tracking + a model registry.
  See `model_notebooks/README.md` for the quickstart and how to point
  at a tracking server.
