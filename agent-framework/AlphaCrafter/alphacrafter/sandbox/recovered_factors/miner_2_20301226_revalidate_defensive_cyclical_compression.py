"""Revalidation only: residual defensive-cyclical dispersion compression loading contraction."""
src=open('scripts/miner_2_20301003_residual_defensive_cyclical_dispersion_compression_loading_contraction_20_60d.py',encoding='utf8').read()
src=src.replace("END=pd.Timestamp('2030-10-02')", "END=pd.Timestamp('2030-12-25')")
src=src.replace("FACTOR residual_defensive_cyclical_dispersion_compression_loading_contraction_20_60d", "FACTOR REVALIDATION_residual_defensive_cyclical_dispersion_compression_loading_contraction_20_60d")
exec(src,globals())
