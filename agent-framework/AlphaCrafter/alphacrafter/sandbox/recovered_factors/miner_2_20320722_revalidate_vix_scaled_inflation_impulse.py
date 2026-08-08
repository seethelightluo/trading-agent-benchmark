"""Revalidation of admitted Miner_2 continuous VIX-scaled inflation impulse through prior completed day."""
src=open('scripts/miner_2_20311030_continuous_vix_scaled_inflation_impulse_residual_loading_contraction_60_20d.py',encoding='utf8').read()
src=src.replace("END=pd.Timestamp('2031-10-29')", "END=pd.Timestamp('2032-07-21')")
src=src.replace('FACTOR continuous_vix_scaled', 'FACTOR REVALIDATION continuous_vix_scaled')
exec(src,globals())
