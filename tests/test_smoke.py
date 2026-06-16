def test_project_loads_config():
    from bolt_pipeliner.config import load_config
    config = load_config('configs/etl_config.yaml')
    assert 'layers' in config
