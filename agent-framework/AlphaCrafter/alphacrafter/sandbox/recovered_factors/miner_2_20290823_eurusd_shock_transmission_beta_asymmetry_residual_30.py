"""One idea validation: EURUSD conditional shock-transmission beta-asymmetry residual, current refresh."""
from pathlib import Path
src=Path('scripts/miner_3_20290614_eurusd_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2029-06-13')", "E=pd.Timestamp('2029-08-22')")
src=src.replace('2029-06-13', '2029-08-22')
exec(compile(src,'miner2_eurusd_shock_refresh_20290823','exec'))
