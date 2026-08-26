import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,5000)
            if x is not None and len(x): return x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
        except Exception: pass
    return None
P={s:load(s) for s in U}; P={s:x for s,x in P.items() if x is not None}
px=pd.DataFrame(P).sort_index().ffill(limit=3); r=np.log(px).diff()
# Adaptive medium-term trend: 60d trend is rewarded when confirmed by 20d trend,
# while cross-asset breadth confidence attenuates signals during internally weak regimes.
m20=r.rolling(20).sum(); m60=r.rolling(60).sum(); vol=r.rolling(40).std().clip(lower=1e-5)
breadth=(r>0).rolling(20).mean().mean(axis=1)
confidence=(breadth-breadth.rolling(120).mean())/breadth.rolling(120).std().clip(lower=1e-6)
confidence=(1/(1+np.exp(-confidence))).clip(.25,.75)
confirm=np.sign(m20)*np.minimum(abs(m20),abs(m60))
f=(.65*m60+.35*confirm).div(vol).mul(confidence,axis=0).shift(1)
fr=px.shift(-20)/px-1
rows=[]
for d in f.index:
    a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
    if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('factor=adaptive_confirmed_breadth_trend20'); print('dates',len(x),'avg_n',x.n.mean(),'coverage',f.notna().sum().sum()/f.size,'IC',m,'ICIR',m/sd*np.sqrt(252),'hit',(x.ic>0).mean()); print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2024'),('2025','2029'),('2030','2034'),('2035','2035')]:
 y=x.loc[a:b].ic; print(a,len(y),y.mean(),y.mean()/y.std(ddof=1)*np.sqrt(252) if len(y)>2 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20351210_adaptive_confirmed_trend_signal.csv',index=False)
