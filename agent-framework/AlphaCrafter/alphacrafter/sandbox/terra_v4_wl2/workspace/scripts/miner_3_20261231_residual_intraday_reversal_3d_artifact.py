import pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-31')
O={}; C={}
for s in U:
    d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
    O[s]=d['open']; C[s]=d['close']
op=pd.concat(O,axis=1,sort=False).reindex(columns=U)
cl=pd.concat(C,axis=1,sort=False).reindex(columns=U)
intr=cl/op-1
res=intr.sub(intr.mean(axis=1),axis=0)
f=-res.rolling(3,min_periods=3).sum()
# Persist date-by-symbol signal artifact; each row uses only completed data through that date.
out=f.rename_axis('date').reset_index().melt(id_vars='date',var_name='symbol',value_name='signal')
out.to_csv('scripts/miner_3_20261231_residual_intraday_reversal_3d_signal.csv',index=False)
print('wrote',len(out),'rows','dates',out.date.nunique(),'valid_signals',int(out.signal.notna().sum()))
