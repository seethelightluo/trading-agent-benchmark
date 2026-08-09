"""Revalidation: 30d VIX-shock transmission beta asymmetry residual through current visible cutoff."""
from pathlib import Path
src=Path('scripts/miner_3_20290322_vix_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2029-03-21')", "E=pd.Timestamp('2029-05-02')")
exec(compile(src,'vix_shock_transmission_revalidation_20290503','exec'))
