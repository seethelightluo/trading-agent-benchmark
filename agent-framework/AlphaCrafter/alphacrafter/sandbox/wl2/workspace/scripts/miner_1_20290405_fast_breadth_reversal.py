import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,days=1800)
 if d is None or len(d)<300: d=get_index_daily_data(s,days=1800)
 return d
raw={s:fetch(s) for s in U}; raw={s:d for s,d in raw.items() if d is not None}
px=pd.DataFrame({s:d.set_index('date').close for s,d in raw.items()})
r=px.pct_change(); vol=r.rolling(12,min_periods=8).std()
# Fast, interpretable reversal: negative 2-session return divided by recent volatility,
# attenuated when the cross-asset median move is strongly one-sided.
base=-r.rolling(2,min_periods=2).sum()/vol
breadth=r.rolling(10,min_periods=6).mean().gt(0).mean(axis=1)
# preserve reversal in balanced markets, reduce it during broad directional regimes
atten=(1-0.55*(2*breadth.sub(0.5).abs()).clip(0,1))
f=base.mul(atten,axis=0).shift(1)
print('universe',len(raw),'rows',len(px),'range',px.index.min(),px.index.max())
for h in [1,5,10,20]:
 fw=px.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z))
 a=np.asarray(vals); print('h%d IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(h,np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),np.mean(a>0),len(a),np.mean(ns)))
rank=f.rank(axis=1,pct=True); print('turnover %.6f coverage %.4f'%(rank.diff().abs().mean(axis=1).mean(),f.notna().sum(axis=1).mean()/len(U)))
fw=px.pct_change().shift(-1)
for yr in range(2021,2030):
 a=[]
 for dt in f.index[f.index.year==yr]:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 if a: print('yr%d IC %.6f ICIR %.4f n%d'%(yr,np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12),len(a)))
# signal artifact for audit
f.to_csv('scripts/miner_1_20290405_fast_breadth_reversal_signal.csv',index_label='date')
