"""Revalidation: residual positive-jump concentration expansion, 20d vs 60d.
Uses completed observations only through 2031-07-09."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20310417_residual_positive_jump_concentration_expansion_20_60d.py',encoding='utf8').read()
# Retain the complete, already audited single-factor methodology, changing only
# its visible-data cutoff and output label.
src=src.replace("END=pd.Timestamp('2031-04-16')", "END=pd.Timestamp('2031-07-09')")
src=src.replace("validation_end',END.date()", "revalidation_end',END.date()")
exec(src, globals())
