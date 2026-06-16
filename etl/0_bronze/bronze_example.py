"""Example bronze ETL job — tutorial template.

Read the comments below, then replace the body with your own logic. Once
your YAML entry exists in configs/etl_config.yaml, run it with:

    python main.py --bronze          # only this layer
    python main.py                  # all layers, in dependency order

For this module to actually execute, configs/etl_config.yaml must have an
entry under `bronze:` whose `module:` value matches this filename (without
`.py`) and whose `input_tables:` block declares the aliases this function reads.
"""
import datetime as dt
import polars as pl


def process_data(self, input_tables):
    """Transform `input_tables` and return a polars.DataFrame.

    Parameters
    ----------
    self : ETLBase
        The runtime instance — your function gets monkey-patched onto it,
        so `self` exposes everything the base class set up. Useful bits:
          • self.spark              SparkSession (Spark bases only)
          • self.incremental_column Incremental column name from YAML.
          • self.incremental_policy Incremental mode settings
                                    (window/append/overwrite).
          • self.partition_by      Echo of the YAML `partition_by:` list.
          • self.incremental       Echo of the YAML `incremental:` flag.
          • self._create_table(df) / self._replace_table_partitions(df)
                                    Manual write helpers — pair with
                                    `unload: false` for memory-heavy jobs.

    input_tables : dict[str, polars.DataFrame]
        **THIS IS NOT YOUR JOB'S INPUT FILES.** It is a dict of
        already-loaded DataFrames, keyed by the *aliases you declared in
        etl_config.yaml* under this job's `input_tables:` block. ETLBase
        loaded each value before calling you.

        Example YAML:

            bronze:
              - module: bronze_example
                input_tables:
                  raw_orders: bronze_orders          # ← alias: source
                  customers:  silver_dim_customers
                output_table_name: example

        At runtime, `input_tables` becomes:

            {
                "raw_orders": <DataFrame loaded from bronze_orders>,
                "customers":  <DataFrame loaded from silver_dim_customers>,
            }

        Where the values come from depends on the layer:
        Bronze values can take two shapes:
          • `"<schema>.<table>"` (contains a dot) → read from the project's
            *shared* catalog (raw ingestion zone, e.g. `raw.crm_account`).
          • `"<bare_table>"` → read from the project's own catalog like a
            silver/gold job. Useful for chained bronze refinements.

    Returns
    -------
    polars.DataFrame
        Persisted by ETLBase to `bronze_<output_table_name>` (partitioned
        per `partition_by:`). If `unload: false` is set in the YAML, write
        partitions yourself via `self._create_table` /
        `self._replace_table_partitions` and return an empty DataFrame
        instead — ETLBase will skip its own unload step.
    """
    # 1. Pick a table by its declared YAML alias. The example below grabs
    #    whichever table happens to be first — in a real job you should
    #    reference aliases explicitly:
    #        orders = input_tables["raw_orders"]
    #        customers = input_tables["customers"]
    df = next(iter(input_tables.values()))

    # 2. (Optional) If you need manual custom incremental handling,
    #    inspect `self.incremental_policy` and `self.incremental_column`.
    #    Built-in bases already apply incremental write modes from YAML.

    # 3. Transform. Below: tag every row with the run timestamp.
    #    Real jobs use .group_by / .join / .with_columns chains.
    return df.with_columns(
        pl.lit(dt.datetime.utcnow()).alias("processed_at")
    )
