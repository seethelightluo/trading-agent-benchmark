import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=3000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index().ffill()
r=np.log(P).diff()
# lagged signal: medium momentum, rewarding compressed volatility and penalizing shocks
mom=np.log(P/P.shift(20)); v10=r.rolling(10).std(); v40=r.rolling(40).std()
f=(mom/(v40+1e-8))*(1-(v10/(v40+1e-8)-1).clip(-1,2)/3)
# ensure each date cross-section and forward return
for h in [5,10,20]:
    ics=[]; n=[]; turns=[]
    for i in range(45,len(P)-h):
        a=f.iloc[i]; y=np.log(P.iloc[i+h]/P.iloc[i]); z=pd.concat([a,y],axis=1).dropna()
        if len(z)>=8:
            ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); n.append(len(z))
            turns.append((a.rank()-f.iloc[i-1].rank()).abs().mean()/len(U))
    q=pd.Series(ics).dropna(); print(h,'dates',len(q),'avgN',np.mean(n),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',np.mean(turns),'coverage',P.notna().mean().mean())
    for w in [365,730,1095]:
        x=q.tail(w); print(' recent',w,x.mean(),x.mean()/x.std(ddof=1))
