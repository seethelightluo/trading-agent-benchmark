"""Revalidation only: admitted signed-linear residual tail-severity/inverse-recovery factor."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20330303_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d.py',encoding='utf8').read()
# Retain the candidate logic and validation protocol; update only visibility-safe cutoff.
text=src.replace("END=pd.Timestamp('2033-03-02')","END=pd.Timestamp('2033-03-30')")
text=text.replace("signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d validation_end","REVALIDATION signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d validation_end")
exec(text,globals())
# Persist aligned signal solely for reproducible revalidation audit.
f.to_pickle('scripts/miner_1_20330331_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d_revalidation_signal.pkl')
