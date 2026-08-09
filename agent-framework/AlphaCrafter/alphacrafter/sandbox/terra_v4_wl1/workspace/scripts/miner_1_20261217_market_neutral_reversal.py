import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
allr={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].set_index('date'); allr[s]=d.close.pct_change()
r=pd.DataFrame(allr); m=r['SPX']; beta=r.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0)
ret=r.rolling(5,min_periods=5).sum(); d=pd.DataFrame(index=r.index)
# negative five-day market-neutral residual, lagged one day
for s in U: d[s]=-(ret[s]-beta[s]*ret['SPX']).shift(1)
x=d.stack().rename('factor').reset_index().rename(columns={'level_0':'date','level_1':'symbol'})
for h in [1,5,10]:
 yy=(r.shift(-h).rolling(h,min_periods=h).sum()).stack().rename('y').reset_index(); yy.columns=['date','symbol','y']; x=x.merge(yy,on=['date','symbol'])
x.to_csv('scripts/miner_1_20261217_market_neutral_reversal_signal.csv',index=False)
for h in [1,5,10]:
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor', 'y']);
  if len(g)>=8 and g.factor.nunique()>1 and g.y.nunique()>1:
   q=spearmanr(g.factor,g.y).statistic
   if np.isfinite(q): a.append((dt,q,len(g)))
 z=pd.DataFrame(a,columns=['date','ic','n']); q=z.ic; print(f'H{h} dates={len(q)} avgN={z.n.mean():.2f} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
 if h==1:
  for lab,gg in [('pre2024',z[z.date<'2024-01-01']),('2024+',z[z.date>='2024-01-01']),('2026+',z[z.date>='2026-01-01'])]: print(lab,len(gg),gg.ic.mean(),gg.ic.mean()/gg.ic.std(ddof=1))
v=x.factor.notna(); print('coverage',v.mean(),'turnover',x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
