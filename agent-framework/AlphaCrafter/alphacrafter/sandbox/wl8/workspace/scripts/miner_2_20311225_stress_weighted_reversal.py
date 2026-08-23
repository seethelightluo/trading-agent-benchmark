import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s,n=4000):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            d=fn(s,n)
            if d is not None and len(d):
                d=d.copy(); d['date']=pd.to_datetime(d['date']); return d.set_index('date')['close'].astype(float)
        except Exception: pass
    return None
ser={s:get(s) for s in U}; px=pd.DataFrame(ser).sort_index().ffill()
try:
    d=get_index_daily_data('VIX',4000); v=d.set_index(pd.to_datetime(d.date))['close'].astype(float)
except Exception: v=pd.Series(20.,index=px.index)
v=v.reindex(px.index).ffill(); r=px.pct_change()
# Stress-weighted short-term reversal: lagged 5d reversal, continuously stronger as VIX percentile rises.
vp=v.rolling(252,min_periods=100).rank(pct=True).shift(1)
f=-px.shift(1).pct_change(5).mul((0.5+vp).clip(0.5,1.5),axis=0)
fr=px.pct_change(10).shift(-10); rows=[]
for dt in f.index:
    x,y=f.loc[dt],fr.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8: rows.append((dt,x[ok].corr(y[ok]),ok.sum()))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(ic),'avg_n',round(ic.n.mean(),3),'coverage',round(ic.n.mean()/15,4))
print('IC %.8f ICIR %.8f hit %.4f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean()))
for w in (252,504,756):
 z=ic.tail(w); print('window',w,'n',len(z),'IC',round(z.ic.mean(),8),'ICIR',round(z.ic.mean()/z.ic.std(),5))
for h in (1,3,5,10,20):
 yy=px.pct_change(h).shift(-h); a=[]
 for dt in f.index:
  x,y=f.loc[dt],yy.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:a.append(x[ok].corr(y[ok]))
 print('decay',h,'IC',round(float(np.nanmean(a)),8),'dates',len(a))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
f.to_csv('scripts/miner_2_20311225_stress_weighted_reversal_signal.csv'); ic.to_csv('scripts/miner_2_20311225_stress_weighted_reversal_ic.csv')
