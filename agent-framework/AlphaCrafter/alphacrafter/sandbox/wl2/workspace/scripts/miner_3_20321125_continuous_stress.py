import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,5000)
            if x is not None and len(x)>100:return x
        except Exception: pass
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items()}).sort_index().groupby(level=0).last()
R=C.ffill().pct_change(); res=R.sub(R.mean(axis=1),axis=0)
v=get_index_daily_data('VIX',5000); V=v.set_index(pd.to_datetime(v.date)).close.astype(float).reindex(C.index).ffill()
shock=res.rolling(3,min_periods=3).sum().shift(1)
breadth=(R<0).mean(axis=1).rolling(3,min_periods=3).mean().shift(1)
vi=V.pct_change(3).shift(1); base=vi.rolling(252,min_periods=100).median().shift(1); scale=vi.rolling(252,min_periods=100).std().shift(1)
# continuous stress intensity: breadth excess over neutral and positive standardized VIX impulse
stress=((breadth-.50).clip(lower=0)/.50) * ((vi-base)/scale).clip(lower=0).clip(upper=4)
vol=R.rolling(20,min_periods=10).std().shift(1)
f=(-shock/vol).mul(stress,axis=0).replace([np.inf,-np.inf],np.nan)
rows_by_h={}
for h in [1,3,5,10]:
    fr=R.rolling(h).sum().shift(-h); rows=[]
    for d in f.index:
        q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
        if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
    a=pd.Series(rows).dropna(); rows_by_h[h]=a
    print('H',h,'dates',len(a),'mean',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
# regime halves and diagnostics
x=rows_by_h[1]; print('assets',len(D),'calendar_dates',len(C),'nonzero_dates',int((stress>0).sum()),'coverage',round(f.notna().sum(axis=1).replace(0,np.nan).mean()/len(D),4))
for label,idx in [('early',x.index < len(x)//2),('late',x.index >= len(x)//2)]: print(label,'n',int(idx.sum()),'ic',round(x[idx].mean(),6))
f.to_csv('scripts/miner_3_20321125_continuous_stress_signal.csv',index_label='date')
