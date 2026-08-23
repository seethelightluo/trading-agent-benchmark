import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-05-03')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']);P[s]=x[x.date<=END].set_index('date').sort_index()
idx=sorted(set.intersection(*[set(v.index) for v in P.values()])); op=pd.DataFrame({s:P[s].reindex(idx).open for s in U}); cl=pd.DataFrame({s:P[s].reindex(idx).close for s in U})
# Lagged overnight gap reversal, normalized by prior 20d volatility.
gap=(op/cl.shift(1)-1); vol=cl.pct_change().rolling(20,min_periods=10).std(); sig=(-gap/vol).shift(1); fwd=cl.shift(-1)/cl-1
rows=[]
for d in idx:
 g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): rows.append((d,q,len(g)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=z.ic
print('candidate=vol-normalized overnight gap reversal');print('dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(sig.notna().sum().sum()/sig.size,4));print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
for lab,m in [('2020-22',z.index<'2023-01-01'),('2023-25',(z.index>='2023-01-01')&(z.index<'2026-01-01')),('2026',(z.index>='2026-01-01')&(z.index<'2027-01-01')),('2027', (z.index>='2027-01-01')&(z.index<'2028-01-01')),('2028',z.index>='2028-01-01'),('recent180',z.index>=END-pd.Timedelta(days=180))]:
 q=a[m];print(lab,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)) if len(q)>1 else '')
for h in [2,5,10]:
 t=cl.shift(-h)/cl-1; qs=[]
 for d in idx:
  g=pd.DataFrame({'s':sig.loc[d],'f':t.loc[d]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: qs.append(spearmanr(g.s,g.f).statistic)
 q=pd.Series(qs).dropna();print('h',h,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20280504_gapvol_signal.csv',index=False)
