import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    d=get_stock_daily_data(s,3100)
    if d is not None and len(d)>=120: D[s]=d.set_index('date')
C=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index(); R=C.pct_change()
r15=C/C.shift(15)-1
res=r15.sub(r15.mean(axis=1),axis=0)
vol=R.rolling(30,min_periods=15).std()*np.sqrt(20)
# Market breadth is causal: fraction of assets with positive trailing 5d return.
breadth=(R.rolling(5,min_periods=5).sum()>0).mean(axis=1)
# Strongly bearish breadth receives a modest boost to reversal, bullish breadth a mild dampener.
# The cross-sectional residual construction prevents common market direction leakage.
gate=(1.25-0.50*breadth).clip(0.75,1.25)
sig=(-res/(vol+1e-8)).mul(gate,axis=0)
print('assets',len(C.columns),'rows',len(C),'start',C.index.min().date(),'end',C.index.max().date())
for h in [5,10,20]:
    q=C.shift(-h)/C-1; vals=[]; ds=[]; ns=[]
    for dt in sig.index:
        z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
            if np.isfinite(c): vals.append(c); ds.append(dt); ns.append(len(z))
    a=np.array(vals); ds=pd.DatetimeIndex(ds); ns=np.array(ns)
    if len(a):
        print('h',h,'dates',len(a),'meanN',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),6))
        if h==10:
            for aa,bb in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2029-12-31'),('2030-01-01','2035-12-19')]:
                w=a[(ds>=aa)&(ds<=bb)]; print('regime',aa,bb,'n',len(w),'ic',round(w.mean(),6) if len(w) else None,'icir',round(w.mean()/w.std(ddof=1),6) if len(w)>1 else None)
            ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds)
            print('turnover',round(ranks.diff().abs().mean().mean(),6))
            pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20351220_breadth_gate_signal.csv',index=False)
