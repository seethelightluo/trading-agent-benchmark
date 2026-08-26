import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    d=get_stock_daily_data(s,days=3000)
    if d is not None and len(d)>260: D[s]=d.set_index('date').sort_index()
c=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); r=c.pct_change()
r60=c.pct_change(60); resid=r60.sub(r60.mean(axis=1),axis=0)
vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
r20=c.pct_change(20); res20=r20.sub(r20.mean(axis=1),axis=0)
breadth=(res20>0).mean(axis=1); gate=(0.35+1.3*breadth).clip(0.35,1.65)
# Clean sign-reversed breadth-gated trend: contrarian residual momentum, inverse volatility scaled.
sig=(-(resid/vol).mul(gate,axis=0)).shift(1)
rows=[]
for h in [5,10,20,40,60]:
    fwd=c.shift(-h)/c-1; a=[]
    for dt in c.index:
        z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)>=8:
            q=z.iloc[:,0].corr(z.iloc[:,1])
            if np.isfinite(q): a.append((dt,q,len(z)))
    x=pd.DataFrame(a,columns=['date','ic','n']); ic=x.ic
    print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/len(U),4),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
    if h==20:
        for n,a1,b1 in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-12-31')]:
            y=x[(x.date>=a1)&(x.date<=b1)]; print('REG',n,len(y),round(y.ic.mean(),6),round(y.ic.mean()/y.ic.std(ddof=1),6),round((y.ic>0).mean(),4))
        print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20310918_breadth_contrarian_signal.csv',index=False)
print('UNIVERSE',len(D),'DATES',len(c),'SIGNAL_ROWS',len(out))
