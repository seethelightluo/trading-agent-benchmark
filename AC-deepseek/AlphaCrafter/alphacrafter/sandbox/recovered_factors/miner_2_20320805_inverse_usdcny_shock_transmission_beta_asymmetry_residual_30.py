"""Miner_2 single-candidate validation: inverse USDCNY shock transmission beta asymmetry."""
from pathlib import Path
src=Path('scripts/miner_3_20320624_inverse_eurusd_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2032-06-23')", "E=pd.Timestamp('2032-08-04')")
src=src.replace("EURUSD daily return is a liquid global-dollar/risk-appetite shock", "USDCNY daily return is a China-growth/renminbi-risk transmission shock")
src=src.replace("EURUSD separately on euro-strength\n# and euro-weakness sessions", "USDCNY separately on CNY-weakness\n# and CNY-strength sessions")
src=src.replace("fx=rd('EURUSD',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)", "fx=rd('USDCNY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)")
src=src.replace('inverse_eurusd_shock_transmission_beta_asymmetry_residual_30','inverse_usdcny_shock_transmission_beta_asymmetry_residual_30')
src=src.replace('miner_3_eurusd_shock_20320624','miner_2_usdcny_shock_20320805')
exec(compile(src,'miner_2_usdcny_shock_20320805','exec'))
