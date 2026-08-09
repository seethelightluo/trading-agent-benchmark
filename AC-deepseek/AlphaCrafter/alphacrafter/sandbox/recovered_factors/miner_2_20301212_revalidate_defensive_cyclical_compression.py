"""Revalidation only: residual defensive-cyclical dispersion compression loading contraction."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_2_20301003_residual_defensive_cyclical_dispersion_compression_loading_contraction_20_60d.py',encoding='utf8').read()
# Use exact prior definition with completed-data cutoff at the runtime prior session.
src=src.replace("END=pd.Timestamp('2030-10-02')", "END=pd.Timestamp('2030-12-11')")
src=src.replace("FACTOR residual_defensive_cyclical_dispersion_compression_loading_contraction_20_60d", "FACTOR REVALIDATION_residual_defensive_cyclical_dispersion_compression_loading_contraction_20_60d")
exec(src,globals())
