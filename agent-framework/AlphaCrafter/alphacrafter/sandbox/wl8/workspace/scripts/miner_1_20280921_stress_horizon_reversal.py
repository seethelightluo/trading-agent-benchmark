import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-09-20')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@END').set_index('date').close.sort_index() for s in U}; px=pd.DataFrame(P).sort_index(); r=px.pct_change();
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@END').set_index('date').close.sort_index().reindex(px.index).ffill(); med=vix.shift(1).rolling(252,min_periods=100).median(); stress=(vix.shift(1)>med).astype(float)
# short reversal, with stress-dependent horizon: 5d reversal in calm, 10d reversal in stress
s=(-r.rolling(5).sum()).shift(1); s10=(-r.rolling(10).sum()).shift(1); sig=s.where(stress<.5,s10)
fw=px.shift(-1)/px-1; rows=[]
for d in px.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fw.loc[d]}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(z),'rows',int(z.n.sum()),'avgN',round(z.n.mean(),2),'coverage',round(sig.notna().sum().sum()/sig.size,4)); print('IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()))
for lab,m in [('2020-22',z.index<'2023-01-01'),('2023-25',(z.index>='2023-01-01')&(z.index<'2026-01-01')),('2026',(z.index>='2026-01-01')&(z.index<'2027-01-01')),('2027',(z.index>='2027-01-01')&(z.index<'2028-01-01')),('2028',z.index>='2028-01-01'),('recent180',z.index>=END-pd.Timedelta(days=180))]:
 q=z[m]; print(lab,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)) if len(q)>1 else 'insufficient')
print('signal artifact written'); sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20280921_stress_horizon_reversal_signal.csv',index=False)
