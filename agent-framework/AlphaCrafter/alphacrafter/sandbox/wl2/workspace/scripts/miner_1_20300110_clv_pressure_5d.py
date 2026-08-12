import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in A:
 x=get_stock_daily_data(s,days=2600)
 if x is None: continue
 x=x.sort_values('date').drop_duplicates('date').set_index('date'); c=x.close.astype(float); h=x.high.astype(float); l=x.low.astype(float); o=x.open.astype(float)
 rng=(h-l).replace(0,np.nan); clv=((c-l)-(h-c))/rng; body=(c-o)/rng
 # Reversal: fade persistent close-location and candle-body pressure, volatility scaled.
 pressure=.65*clv.rolling(5).mean()+.35*body.rolling(5).mean()
 f=-pressure/(c.pct_change().rolling(20).std()+.01)
 for i in range(25,len(x)-10):
  if np.isfinite(f.iloc[i]): rows.append((x.index[i],s,f.iloc[i],c.iloc[i+1]/c.iloc[i]-1,c.iloc[i+5]/c.iloc[i]-1))
z=pd.DataFrame(rows,columns=['date','symbol','signal','r1','r5']); z[['date','symbol','signal']].to_csv('scripts/miner_1_20300110_clv_pressure_5d_signal.csv',index=False)
print('rows',len(z),'assets',z.symbol.nunique())
for k in [1,5]:
 ic=[]; ns=[]
 for d,g in z.groupby('date'):
  g=g.dropna(subset=['signal','r'+str(k)])
  if len(g)>=8: ic.append(g.signal.corr(g['r'+str(k)],method='spearman')); ns.append(len(g))
 a=np.array(ic,float); print('h%d IC %.5f ICIR %.5f hit %.4f dates %d avg_n %.2f coverage %.4f'%(k,np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),len(a),np.mean(ns),np.mean(ns)/15))
for label,lo,hi in [('early','2020-01-01','2025-12-31'),('late','2026-01-01','2030-01-10')]:
 q=z[(z.date>=lo)&(z.date<=hi)]; ic=[]
 for d,g in q.groupby('date'):
  if len(g)>=8: ic.append(g.signal.corr(g.r1,method='spearman'))
 a=np.array(ic,float); print(label,'IC %.5f ICIR %.5f hit %.3f dates %d'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),len(a)))
print('turnover proxy',np.mean(z.sort_values(['symbol','date']).groupby('symbol').signal.apply(lambda x: np.mean(np.sign(x).diff().fillna(0)!=0))))
