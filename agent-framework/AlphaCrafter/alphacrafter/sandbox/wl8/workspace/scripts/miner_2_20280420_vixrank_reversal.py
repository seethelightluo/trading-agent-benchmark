import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-04-19')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); fw=px.shift(-1)/px-1
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']);v=v[v.date<=END].set_index('date')['close'].sort_index().reindex(px.index).ffill()
base=(-(r.rolling(5).sum().rank(axis=1,pct=True)-.5)/(0.5+r.rolling(20).std().rank(axis=1,pct=True)))
shock=(v>v.rolling(60,min_periods=40).median()).astype(float)
sig=(base.multiply(shock,axis=0)).shift(1)
rows=[]
for d in px.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fw.loc[d]},index=px.columns).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q):rows.append((d,q,len(g)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('candidate=VIX-high rank-volatility reversal');print('dates',len(z),'avgN',z.n.mean(),'coverage',sig.notna().sum().sum()/sig.size,'assets',len(U));print('IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()))
for lab,m in [('2026',z.index.year==2026),('2027',z.index.year==2027),('2028',z.index.year==2028),('recent180',z.index>=END-pd.Timedelta(days=180))]:
 q=z[m];print(lab,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)) if len(q)>1 else '')
print('turnover_proxy',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20280420_vixrank_reversal_signal.csv',index=False)
