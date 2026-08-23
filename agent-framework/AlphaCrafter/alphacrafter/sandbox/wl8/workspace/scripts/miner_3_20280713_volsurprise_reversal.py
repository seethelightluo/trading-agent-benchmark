import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-07-11')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Volatility-surprise reversal: fade the latest completed return when its magnitude
# is unusually large relative to each asset's own lagged 20d volatility.
vol=r.rolling(20,min_periods=15).std().shift(1)
sig=-(r.shift(1)/vol)
fw={h:px.shift(-h)/px-1 for h in [1,3,5,10]}
allz={}
for h,f in fw.items():
 rows=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); allz[h]=z
 print('h',h,'dates',len(z),'rows',int(z.n.sum()),'avgN',round(z.n.mean(),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()))
 for lab,m in [('2020-22',z.index<'2023-01-01'),('2023-25',(z.index>='2023-01-01')&(z.index<'2026-01-01')),('2026',(z.index>='2026-01-01')&(z.index<'2027-01-01')),('2027',(z.index>='2027-01-01')&(z.index<'2028-01-01')),('2028',z.index>='2028-01-01'),('recent180',z.index>=END-pd.Timedelta(days=180))]:
  q=z[m]
  print(' ',lab,'dates',len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)) if len(q)>1 else ' insufficient')
# turnover: rank ordering changes, measured daily Spearman rank correlation
rank=sig.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rank)):
 a=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(a)>=8: turn.append(1-a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
print('turnover_proxy',np.nanmean(turn),'turnover_obs',len(turn))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20280713_volsurprise_reversal_signal.csv',index=False)
