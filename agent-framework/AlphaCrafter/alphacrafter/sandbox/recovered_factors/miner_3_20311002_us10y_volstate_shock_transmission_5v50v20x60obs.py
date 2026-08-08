"""Current revalidation of one candidate: US10Y volatility-state shock-transmission exposure.
Delegates to the July specification, replacing only the visible-data endpoint (2031-10-01).
"""
from pathlib import Path
src=Path('scripts/miner_3_20310724_us10y_volstate_shock_transmission_5v50v20x60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2031-07-23')", "END=pd.Timestamp('2031-10-01')")
src=src.replace('2031-07-23', '2031-10-01')
src=src.replace('miner_3_20310724_us10y_volstate_shock_transmission_5v50v20x60obs_signal.pkl', 'miner_3_20311002_us10y_volstate_shock_transmission_5v50v20x60obs_signal.pkl')
exec(compile(src, 'yield_transmission_current.py', 'exec'))
