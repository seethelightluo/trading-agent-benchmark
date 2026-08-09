"""Revalidation only: residual commodity-crypto dispersion compression loading expansion."""
# Definition is unchanged from its admitted 2030-09-05 version; all closes end 2030-12-11.
import json, numpy as np, pandas as pd
src=open('scripts/miner_2_20300905_residual_commodity_crypto_dispersion_compression_loading_expansion_20_60d.py',encoding='utf8').read()
src=src.replace("END=pd.Timestamp('2030-09-04')", "END=pd.Timestamp('2030-12-11')")
src=src.replace("FACTOR residual_commodity_crypto_dispersion_compression_loading_expansion_20_60d", "FACTOR REVALIDATION_residual_commodity_crypto_dispersion_compression_loading_expansion_20_60d")
exec(src,globals())
