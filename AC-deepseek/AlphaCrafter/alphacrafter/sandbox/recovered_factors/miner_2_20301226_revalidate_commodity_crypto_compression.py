"""Revalidation only: residual commodity-crypto dispersion compression loading expansion.
Uses completed closes only through 2030-12-25, the day before this cycle date.
"""
src=open('scripts/miner_2_20300905_residual_commodity_crypto_dispersion_compression_loading_expansion_20_60d.py',encoding='utf8').read()
src=src.replace("END=pd.Timestamp('2030-09-04')", "END=pd.Timestamp('2030-12-25')")
src=src.replace("FACTOR residual_commodity_crypto_dispersion_compression_loading_expansion_20_60d", "FACTOR REVALIDATION_residual_commodity_crypto_dispersion_compression_loading_expansion_20_60d")
exec(src,globals())
