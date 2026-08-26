import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').drop_duplicates('date'); F[s]=d
print('assets',len(F),'avg history',np.mean([len(x) for x in F.values()]))
# Candidate: lagged 5-day close-location / range pressure. Each day close location in its
# high-low range, smoothed by range weighting; persistent closes near highs forecast continuation.
rows=[]
for s,d in F.items():
 c=pd.to_numeric(d.close,errors='coerce'); h=pd.to_numeric(d.high,errors='coerce'); l=pd.to_numeric(d.low,errors='coerce'); v=pd.to_numeric(d.volume,errors='coerce')
 rng=(h-l).replace(0,np.nan); loc=(2*c-h-l)/rng
 # range-weighted signed location, robustly clipped; lag prevents lookahead
 sig=(loc.clip(-1,1)*rng/c).rolling(5,min_periods=3).sum()
 sig=sig/(rng/c).rolling(20,min_periods=10).sum().replace(0,np.nan)
 rows.append(pd.DataFrame({'date':d.date,'asset':s,'signal':sig.shift(1),'close':c}))
a=pd.concat(rows).sort_values(['date','asset'])
for H in [1,5,10,20]:
 a['fwd']=a.groupby('asset').close.shift(-H)/a.close-1
 vals=[]
 for dt,g in a.groupby('date'):
  z=g.dropna(subset=['signal','fwd'])
  if len(z)>=8: vals.append((dt,len(z),z.signal.corr(z.fwd,method='spearman')))
 q=pd.DataFrame(vals,columns=['date','n','ic']).dropna(); m=q.ic.mean(); ir=m/q.ic.std(ddof=1)*np.sqrt(252)
 print('H',H,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC %.8f ICIR %.8f hit %.4f'%(m,ir,(q.ic>0).mean()))
 if H==10:
  for name,sub in [('early',q.iloc[:len(q)//3]),('mid',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]: print(name,len(sub),'IC %.8f ICIR %.8f'%(sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1)*np.sqrt(252)))
  q.to_csv('scripts/miner_3_20300617_clv_pressure_ic.csv',index=False)
# coverage and rank turnover
r=a.dropna(subset=['signal']).pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True)
print('coverage %.4f turnover %.6f'%(a.signal.notna().groupby(a.date).mean().mean(),(r.diff().abs().mean(axis=1)/2).dropna().mean()))
a.to_csv('scripts/miner_3_20300617_clv_pressure_signal.csv',index=False)
