"""Revalidation: residual positive-jump concentration expansion, 20d vs 60d.
Visible completed daily observations are limited to 2031-08-06."""
src=open('scripts/miner_1_20310417_residual_positive_jump_concentration_expansion_20_60d.py',encoding='utf8').read()
src=src.replace("END=pd.Timestamp('2031-04-16')", "END=pd.Timestamp('2031-08-06')")
src=src.replace("validation_end',END.date()", "revalidation_end',END.date()")
exec(src, globals())
