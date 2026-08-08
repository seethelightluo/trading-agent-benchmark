"""Scheduled revalidation using information visible through 2033-07-06.
One existing Miner-1 idea only; no new candidate is mixed into this test.
"""
import pandas as pd
src=open('scripts/miner_1_20330303_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d.py',encoding='utf8').read()
text=src.replace("END=pd.Timestamp('2033-03-02')", "END=pd.Timestamp('2033-07-06')")
text=text.replace('signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d validation_end', 'REVALIDATION signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d validation_end')
exec(text, globals())
f.to_pickle('scripts/miner_1_20330707_signed_linear_residual_tail_severity_revalidation_signal.pkl')
