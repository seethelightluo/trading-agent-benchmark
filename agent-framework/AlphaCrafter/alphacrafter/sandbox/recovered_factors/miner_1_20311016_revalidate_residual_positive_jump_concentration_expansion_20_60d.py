"""Scheduled revalidation of one existing residual positive-jump concentration factor.
Uses observations completed through 2031-10-15; no later data are requested."""
src=open('scripts/miner_1_20310417_residual_positive_jump_concentration_expansion_20_60d.py',encoding='utf8').read()
src=src.replace("END=pd.Timestamp('2031-04-16')", "END=pd.Timestamp('2031-10-15')")
src=src.replace("validation_end',END.date()", "revalidation_end',END.date()")
exec(src, globals())
