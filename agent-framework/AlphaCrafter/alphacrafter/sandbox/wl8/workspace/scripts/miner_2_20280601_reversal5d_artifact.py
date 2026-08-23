import os,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d['date']=pd.to_datetime(d.date);p[s]=d.sort_values('date').set_index('date').close
px=pd.concat(p,axis=1).sort_index(); sig=-px.shift(1)/px.shift(6)+1
out=sig.loc[:'2028-05-31'].reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('scripts/miner_2_20280601_reversal5d_signal.csv',index=False)
print('artifact rows',len(out),'dates',out.date.nunique(),'symbols',out.symbol.nunique())
