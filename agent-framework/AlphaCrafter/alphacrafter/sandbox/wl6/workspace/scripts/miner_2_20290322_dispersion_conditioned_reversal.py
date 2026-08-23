import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s,days=3000)
            if x is not None and len(x): return x
        except Exception: pass
raw={s:fetch(s) for s in U}
p=pd.DataFrame({s:x.set_index('date')['close'] for s,x in raw.items() if x is not None}).sort_index()
r=p.pct_change(); m=r.mean(axis=1); res=r.sub(m,axis=0)
rv=res.rolling(20,min_periods=15).std()
# Short-term residual reversal, strengthened when cross-sectional dispersion is elevated.
rev=-res.rolling(5,min_periods=4).sum()/(rv*np.sqrt(5)+1e-12)
disp=res.std(axis=1).rolling(60,min_periods=30).apply(lambda x: (x.iloc[-1]-x.mean())/(x.std(ddof=1)+1e-12),raw=False)
condition=(1+0.35*disp.clip(-1.5,1.5)).clip(0.4,1.6)
f=rev.mul(condition,axis=0)
rank=f.rank(axis=1,pct=True)
print('universe=%d dates=%d'%(len(p.columns),len(p)))
for h in [1,5,10,20]:
    fw=p.shift(-h)/p-1; qs=[]; ns=[]; dates=[]
    for d in f.index:
        z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
        if len(z)>=8:
            qs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(d)
    q=pd.Series(qs,index=pd.to_datetime(dates))
    cov=np.isfinite(f.loc[dates]).sum().sum()/(len(dates)*len(p.columns))
    print('h=%d dates=%d avg_n=%.2f coverage=%.4f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%(h,len(q),np.mean(ns),cov,q.mean(),q.mean()/q.std(ddof=1), (q>0).mean(),rank.diff().abs().mean(axis=1).mean()))
    for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028'),('2029','2030')]:
        z=q[(q.index>=lo)&(q.index<=hi)]
        if len(z): print(' ',lo+'-'+hi,'n=%d IC=%.6f ICIR=%.6f'%(len(z),z.mean(),z.mean()/z.std(ddof=1)))
