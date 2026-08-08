"""One idea validation: refreshed EURUSD conditional shock-transmission beta asymmetry residual."""
from pathlib import Path
src=Path('scripts/miner_3_20290614_eurusd_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2029-06-13')", "E=pd.Timestamp('2029-07-11')")
# Ensure report identity states the current factor/date.
src=src.replace('2029-06-13', '2029-07-11')
exec(compile(src,'eurusd_shock_transmission_refresh_20290712','exec'))
