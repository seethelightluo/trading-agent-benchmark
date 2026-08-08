"""One-idea validation: tail regional equity divergence transmission beta residual."""
from pathlib import Path
src=Path('scripts/miner_1_20290614_tail_regional_equity_divergence_transmission_beta_residual_30.py').read_text()
# Advance only the research cutoff/date declaration. Data APIs retain visible-data discipline.
src=src.replace("E=pd.Timestamp('2029-06-13')", "E=pd.Timestamp('2029-06-27')")
exec(compile(src,'tail_regional_equity_divergence_20290628','exec'))
