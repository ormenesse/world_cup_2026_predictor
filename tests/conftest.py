"""Configuração de path para os testes.

Garante que tanto o projeto (`etl.*`) quanto o pacote vendorizado
(`bolt_pipeliner`, em `_boltpipeliner/`) sejam importáveis ao rodar `pytest`,
independentemente do diretório de invocação.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # .../football_analysis
_VENDOR = _ROOT / "_boltpipeliner"

for p in (_ROOT, _VENDOR):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
