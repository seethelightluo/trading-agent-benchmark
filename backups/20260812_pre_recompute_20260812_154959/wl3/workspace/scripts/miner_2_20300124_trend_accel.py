import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get(s):
    for f in (get_stock_daily_data,get_index_daily_data):
        try:
            x=f(s,days=2600)
            if x is not None and len(x): return x
        except Exception: pass
    return None
px={s:get(s) for s in U}; close=pd.concat({s:x.set_index('date')['close'] for s,x in px.items() if x is not None},axis=1).sort_index()
# Candidate: volatility-adjusted relative trend acceleration: intermediate 20d excess return
# relative to cross-section, normalized by own 20d realized vol; lagged one day.
r=np.log(close).diff(); mom20=np.log(close/close.shift(20)); mom60=np.log(close/close.shift(60))
vol20=r.rolling(20).std()*np.sqrt(20)
# acceleration rewards recent trend versus slow trend, cross-sectional relative and volatility scaled
raw=(mom20-mom60/3)/vol20
sig=raw.sub(raw.median(axis=1),axis=0).shift(1)
# forward returns
outs=[]; artifact=[]
for h in [1,3,5,10]:
    fwd=np.log(close.shift(-h)/close)
    vals=[]
    for d in sig.index:
        a=sig.loc[d]; b=fwd.loc[d]; z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    q=pd.Series(vals).dropna(); outs.append((h,len(q),q.mean(),q.mean()/q.std(ddof=1), (q>0).mean(), len(z)))
# save artifact for daily signal
sig.to_csv('scripts/miner_2_20300124_trend_accel_signal.csv')
print('dates',len(close),'instruments',close.shape[1])
for x in outs: print('h,d,IC,ICIR,hit,lastN',x)
print('coverage',sig.notna().sum(axis=1).mean()/len(U),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
