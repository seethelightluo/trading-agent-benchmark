import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=END].copy(); r=d.close.pct_change()
 # Downside-risk-adjusted medium-term momentum: recent 10d return penalized by recent downside semideviation
 dn=r.clip(upper=0)
 d['factor']=d.close.shift(1).pct_change(10)/(dn.shift(1).rolling(20,min_periods=10).std()+1e-8)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True); x.to_csv('scripts/miner_1_20261217_downside_adj_momentum_signal.csv',index=False)
for h in [1,5,10]:
 out=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>1 and g[f'y{h}'].nunique()>1:
   ic=spearmanr(g.factor,g[f'y{h}']).statistic
   if np.isfinite(ic): out.append((dt,ic,len(g)))
 z=pd.DataFrame(out,columns=['date','ic','n']); q=z.ic
 print(f'H{h} dates={len(q)} avgN={z.n.mean():.2f} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
 if h==1:
  for label, gg in [('pre2024',z[z.date<'2024-01-01']),('2024+',z[z.date>='2024-01-01']),('2026+',z[z.date>='2026-01-01'])]:
   print(label,len(gg),gg.ic.mean(),gg.ic.mean()/gg.ic.std(ddof=1))
v=x.dropna(subset=['factor']); print('coverage',len(v)/len(x),'turnover',v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
