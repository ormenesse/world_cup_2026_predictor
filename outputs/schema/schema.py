import pandas as pd
import io

def get_table_schema(spark, job_script_name, schema, catalog):
    """
    Try to fetch table schema using spark
    """
    try:
        table_schema = spark.sql(f"DESCRIBE {catalog}.{schema}.{job_script_name}")
        table_schema = table_schema.toPandas()
        mask = (
            table_schema["col_name"].notna()
            & (table_schema["col_name"] != "")
            & (~table_schema["col_name"].str.startswith("#"))
        )

        table_schema = table_schema.loc[mask].copy()
        table_schema.loc[:, "table_name"] = job_script_name
    except Exception as e:
        print(f"Error fetching schema: {e}")
        table_schema = pd.DataFrame([],columns=['table_name','col_name','data_type','comment'])
    return table_schema

def return_generated_schema(spark):
    schema = "football_analysis"
    catalog = "dev_catalog"
    schemas = pd.DataFrame([],columns=['table_name','col_name','data_type','comment'])
    tables = ["flatfile_sb_partidas","flatfile_sb_escalacoes","flatfile_sb_stats_jogador","flatfile_sb_competicoes","flatfile_fbref_jogadores_temporada","flatfile_fbref_jogadores_chutes","flatfile_fbref_jogadores_diversos","flatfile_fbref_jogadores_minutos","flatfile_fbref_goleiros","flatfile_fifa_players","flatfile_fifa_schedule","bronze_partidas","bronze_escalacoes","bronze_fbref_partidas","bronze_fbref_escalacoes","bronze_fbref_stats_jogador","bronze_fifa_players","bronze_fifa_matches","silver_jogador_year_month","silver_time_year_month","silver_time_defensivo","silver_jogador_defensivo","silver_fifa_team_snapshot","silver_fifa_match_snapshot","gold_partidas_features","gold_fifa_partidas"]
    for table in tables:
        table_schema = get_table_schema(
                            spark, table, schema,
                            catalog)
        if len(table_schema) > 0:
            schemas = pd.concat([schemas,table_schema],axis=0)
    buffer = io.StringIO()
    schemas.to_csv(buffer, index=False, header=True)
    return schemas, buffer.getvalue()

schemas, str_schemas = return_generated_schema(spark)
print("-"*60)
print("Please, copy and paste the following output at: ./outputs/schema/schema.csv\n")
print("-"*60+"\n")
print(str_schemas)