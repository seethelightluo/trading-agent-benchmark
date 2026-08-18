import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(sym):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(sym,days=5000)
            if x is not None and len(x)>100: return x[['date','close']].copy()
        except Exception: pass
    return None
xs=[]
for s in U:
    x=get(s)
    if x is not None:
        x['r']=x.close.pct_change(1); x['r5']=x.close.pct_change(5); x['vol20']=x.r.rolling(20).std()
        xs.append(x.set_index('date').rename(columns={'r':s+'_r','r5':s+'_r5','vol20':s+'_v'}))
d=pd.concat(xs,axis=1).sort_index()
rets=d[[s+'_r' for s in U]].rename(columns=lambda z:z[:-2])
r5=d[[s+'_r5' for s in U]].rename(columns=lambda z:z[:-3])
vol=d[[s+'_v' for s in U]].rename(columns=lambda z:z[:-2])
# signal is negative 5d residual vs cross-sectional median, scaled by own vol; lag one day
res=r5.sub(r5.median(axis=1),axis=0)
f=-(res/(vol* np.sqrt(5)+1e-8)).shift(1)
# forward compounded returns
for h in [1,5,10,20]:
    fr=(1+rets).rolling(h).apply(np.prod,raw=True).shift(-h+1)-1
    vals=[]; nms=[]
    for dt in f.index:
        a=f.loc[dt]; b=fr.loc[dt]
        z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); nms.append(len(z))
    q=pd.Series(vals)
    print(f'H{h}: dates={len(q)} avg_n={np.mean(nms):.2f} min_n={min(nms) if nms else 0} IC={q.mean():.6f} ICIR={(q.mean()/q.std(ddof=1)) if q.std(ddof=1)>0 else np.nan:.6f} hit={(q>0).mean():.4f} recent250={q.tail(250).mean():.6f}/{(q.tail(250).mean()/q.tail(250).std(ddof=1)) if q.tail(250).std(ddof=1)>0 else np.nan}')
print('coverage=',f.notna().sum(axis=1).mean()/15,'dates=',len(f),'assets=',len(xs))
# turnover: rank correlation adjacent valid cross-sections
cs=[]
for dt in f.index:
 a=f.loc[dt].dropna()
 if len(a)>=8: cs.append(a)
trs=[]
for a,b in zip(cs[:-1],cs[1:]):
 z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8: trs.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('turnover_proxy=',np.mean(trs))
