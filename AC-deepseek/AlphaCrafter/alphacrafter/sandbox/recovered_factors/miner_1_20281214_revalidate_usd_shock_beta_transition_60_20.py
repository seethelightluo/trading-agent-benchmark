"""Refresh exact residualized USD-shock beta-transition 60/20 through 2028-12-13."""
import pathlib
src=pathlib.Path('scripts/miner_1_20281130_residualized_usd_shock_beta_transition_60_20.py').read_text().replace("END=pd.Timestamp('2028-11-29')","END=pd.Timestamp('2028-12-13')")
exec(src)
