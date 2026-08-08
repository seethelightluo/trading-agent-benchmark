"""Miner_2 single-candidate validation: inverse USDJPY shock transmission beta asymmetry."""
from pathlib import Path
src=Path('scripts/miner_3_20320624_inverse_eurusd_shock_transmission_beta_asymmetry_residual_30.py').read_text()
# retain the complete established visible-data validation / library-novelty harness,
# changing only the observation-only macro shock and candidate definition.
src=src.replace("E=pd.Timestamp('2032-06-23')", "E=pd.Timestamp('2032-07-21')")
src=src.replace("EURUSD daily return is a liquid global-dollar/risk-appetite shock", "USDJPY daily return is a liquid yen-funding/risk-regime shock")
src=src.replace("EURUSD separately on euro-strength\n# and euro-weakness sessions", "USDJPY separately on yen-weakness\n# and yen-strength sessions")
src=src.replace("fx=rd('EURUSD',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)", "fx=rd('USDJPY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)")
src=src.replace('inverse_eurusd_shock_transmission_beta_asymmetry_residual_30','inverse_usdjpy_shock_transmission_beta_asymmetry_residual_30')
src=src.replace('miner_3_eurusd_shock_20320624','miner_2_usdjpy_shock_20320722')
exec(compile(src,'miner_2_usdjpy_shock_20320722','exec'))
