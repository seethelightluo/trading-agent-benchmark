"""Quarterly revalidation through last completed session before 2031-06-26."""
src=open('scripts/miner_3_20290405_vix_stress_residual_downside_coskewness_contraction_20_60d.py',encoding='utf8').read()
src=src.replace("END=pd.Timestamp('2029-04-04')", "END=pd.Timestamp('2031-06-25')")
src=src.replace("FACTOR vix_stress_residual_downside_coskewness_contraction_20_60d", "FACTOR REVALIDATION_vix_stress_residual_downside_coskewness_contraction_20_60d")
exec(compile(src, 'vix_coskew_revalidation_20310626_body.py', 'exec'))
