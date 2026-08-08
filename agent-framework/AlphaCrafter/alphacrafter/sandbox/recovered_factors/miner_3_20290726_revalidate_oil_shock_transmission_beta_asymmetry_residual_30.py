"""Quarterly revalidation: oil-shock transmission beta-asymmetry residual, visible 2029-07-25."""
from pathlib import Path
src=Path('scripts/miner_3_20290419_oil_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2029-04-18')", "E=pd.Timestamp('2029-07-25')")
# This is a revalidation of the same admitted definition, retaining all historical IC diagnostics.
exec(compile(src,'oil_shock_transmission_revalidation','exec'))
