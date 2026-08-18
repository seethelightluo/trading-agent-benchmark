import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in assets:
    try: x=get_index_daily_data(s, days=4000)
    except Exception: x=get_stock_daily_data(s, days=4000)
    if x is not None and len(x):
        z=x.copy(); z['date']=pd.to_datetime(z['date']); D[s]=z.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index().ffill()
# return and factor available at end of t-1: calculate from prices at t-1, then align factor to date t
r=P.pct_change()
res=r.rolling(10,min_periods=10).sum().sub(r.rolling(10,min_periods=10).sum().median(axis=1),axis=0)
vol=r.rolling(40,min_periods=30).std()*np.sqrt(252)
dd=P/P.rolling(60,min_periods=50).max()-1
# broad, continuous and valid for all assets once windows exist
F=-(res/(vol+1e-8))*(1+0.75*np.clip(-dd,0,0.5))
F=F.shift(1)
# forward return from decision date close to close H sessions later
for H in [5,10,20]:
    fr=P.shift(-H)/P-1
    vals=[]; ns=[]; dates=[]
    for dt in P.index:
        a=F.loc[dt]; b=fr.loc[dt]; q=pd.concat([a,b],axis=1).dropna()
        if len(q)>=8:
            vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q)); dates.append(dt)
    ic=np.asarray(vals); mean=ic.mean(); sd=ic.std(ddof=1)
    print(f'H{H} dates={len(ic)} avgN={np.mean(ns):.2f} IC={mean:.6f} ICIR={mean/sd*np.sqrt(252/H):.6f} hit={np.mean(ic>0):.4f} coverage={np.mean(ns)/len(assets):.4f}')
    if H==10:
      for n in [365,730,1095]:
        z=ic[-n:] if len(ic)>=n else ic
        print(f' recent{min(n,len(ic))} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1)*np.sqrt(252/H):.6f}')
# simple rank turnover at 10d sampling
ranks=F.rank(axis=1,pct=True); common=ranks.dropna(how='all')
turn=[]
for i in range(1,len(common)):
    if i%10==0: turn.append(np.mean(np.abs(common.iloc[i]-common.iloc[i-1]).dropna()))
print('instruments',len(D),'dates',len(P),'factor_valid_rows',F.dropna(how='all').shape[0],'turnover10',np.mean(turn))
