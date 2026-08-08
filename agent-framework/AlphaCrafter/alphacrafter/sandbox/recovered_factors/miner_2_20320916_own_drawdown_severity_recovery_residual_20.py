"""Single idea: own drawdown severity recovery residual, refreshed endpoint."""
from pathlib import Path
src=Path('scripts/miner_2_20320415_own_drawdown_severity_recovery_residual_20.py').read_text()
src=src.replace("E=pd.Timestamp('2032-04-14')", "E=pd.Timestamp('2032-09-15')")
exec(compile(src,'miner_2_20320916_own_drawdown_severity_recovery_residual_20','exec'))
