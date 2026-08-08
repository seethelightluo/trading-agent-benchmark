"""Miner_2 scheduled revalidation: continuous VIX-surprise transmission beta residual, visible through 2029-08-08."""
from pathlib import Path
src=Path('scripts/miner_2_20290503_continuous_vix_surprise_transmission_beta_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2029-05-02')", "E=pd.Timestamp('2029-08-08')")
src=src.replace('through 2029-05-02', 'through 2029-08-08')
exec(compile(src,'miner_2_20290809_revalidate_continuous_vix_surprise','exec'))
