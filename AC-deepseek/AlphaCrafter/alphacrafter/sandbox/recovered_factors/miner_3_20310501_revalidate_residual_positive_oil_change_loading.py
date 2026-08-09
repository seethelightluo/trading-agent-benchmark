"""Scheduled revalidation only: residual positive oil-change shock loading contraction, cutoff 2031-04-30."""
src=open('scripts/miner_3_20300822_residual_positive_oil_change_shock_loading_contraction_20_60d.py',encoding='utf8').read()
src=src.replace("END=pd.Timestamp('2030-08-21')", "END=pd.Timestamp('2031-04-30')")
exec(compile(src, 'oil_revalidation_20310501_body.py', 'exec'))
