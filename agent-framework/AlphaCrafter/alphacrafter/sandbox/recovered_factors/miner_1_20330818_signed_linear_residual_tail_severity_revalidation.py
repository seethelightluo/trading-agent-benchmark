"""Scheduled revalidation of one admitted Miner-1 factor through latest completed bar."""
import os
import pandas as pd
# Derive conservative latest completed common price date, then reuse original reproducible definition.
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ends=[]
for a in assets:
    d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date'])['date']
    ends.append(d.max())
end=min(ends)
print('LATEST_COMMON_COMPLETED_DATE',end.date())
src=open('scripts/miner_1_20330303_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d.py',encoding='utf8').read()
src=src.replace("END=pd.Timestamp('2033-03-02')", "END=pd.Timestamp('%s')" % end.date())
src=src.replace('signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d validation_end', 'REVALIDATION signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d validation_end')
exec(src,globals())
f.to_pickle('scripts/miner_1_20330818_signed_linear_residual_tail_severity_revalidation_signal.pkl')
