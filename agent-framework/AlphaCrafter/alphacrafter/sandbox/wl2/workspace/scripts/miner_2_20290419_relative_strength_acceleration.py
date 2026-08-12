import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=4000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=np.log(px).diff()
# Relative-strength acceleration: recent return momentum relative to its own prior trend,
# scaled by volatility; all inputs lagged one completed session.
for a,b in [(5,20),(10,40),(20,60),(10,60)]:
    mom=r.rolling(a,min_periods=a).sum()-r.rolling(b,min_periods=b).sum()
    vol=r.rolling(20,min_periods=10).std()*np.sqrt(252)
    sig=(mom/(vol+1e-8)).shift(1)
    for h in [1,3,5,10]:
        fwd=px.pct_change(h).shift(-h); rows=[]
        for dt in sig.index:
            z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
            if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
                rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
        q=pd.DataFrame(rows,columns=['date','ic','n']); x=q.ic.to_numpy()
        if h==1:
            yr=q.assign(year=q.date.dt.year).groupby('year').ic.mean().round(4).to_dict()
        else: yr={}
        print('VAR',a,b,'H',h,'dates',len(x),'avgN',round(q.n.mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),3),'coverage',round(q.n.mean()/15,3),'years',yr)
