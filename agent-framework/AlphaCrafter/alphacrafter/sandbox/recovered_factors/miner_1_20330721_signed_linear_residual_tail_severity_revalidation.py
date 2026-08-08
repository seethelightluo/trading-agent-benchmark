"""Scheduled revalidation through 2033-07-20; one existing idea only."""
src=open('scripts/miner_1_20330303_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d.py',encoding='utf8').read()
text=src.replace("END=pd.Timestamp('2033-03-02')", "END=pd.Timestamp('2033-07-20')")
text=text.replace('signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d validation_end', 'REVALIDATION signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d validation_end')
exec(text, globals())
f.to_pickle('scripts/miner_1_20330721_signed_linear_residual_tail_severity_revalidation_signal.pkl')
