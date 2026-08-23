import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-04-19')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); fw=px.shift(-1)/px-1
# Rank-based risk adjusted reversal. Lag all inputs one completed day.
ret5=r.rolling(5).sum(); vol20=r.rolling(20).std()
rk=ret5.rank(axis=1,pct=True); vk=vol20.rank(axis=1,pct=True)
sig=(-(rk-0.5)/(0.5+vk)).shift(1)
rows=[]
for d in px.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fw.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): rows.append((d,q,len(g)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=rank-volatility-scaled 5d reversal');print('dates',len(z),'avgN',z.n.mean(),'coverage',sig.notna().sum().sum()/sig.size,'assets',len(U))
print('IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()))
for lab,m in [('2020-22',z.index<'2023-01-01'),('2023-25',(z.index>='2023-01-01')&(z.index<'2026-01-01')),('2026',(z.index>='2026-01-01')&(z.index<'2027-01-01')),('2027+',z.index>='2027-01-01'),('recent180',z.index>=END-pd.Timedelta(days=180))]:
 q=z[m]; print(lab,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)) if len(q)>1 else '')
for h in [3,5,10]:
 t=px.shift(-h)/px-1;a=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':t.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:a.append(spearmanr(g.s,g.f).statistic)
 q=pd.Series(a).dropna();print('h',h,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('turnover_proxy',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20280420_rankvol_reversal_signal.csv',index=False)
