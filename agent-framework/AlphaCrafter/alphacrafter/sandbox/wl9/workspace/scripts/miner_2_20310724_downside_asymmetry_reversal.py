import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    d=get_stock_daily_data(s,days=3000)
    if d is not None and len(d)>220: D[s]=d.set_index('date').sort_index()
c=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index()
r=c.pct_change(); csmean=r.mean(axis=1); resid=r.sub(csmean,axis=0)
res60=resid.rolling(60,min_periods=40).sum()
down=resid.clip(upper=0).pow(2).rolling(40,min_periods=20).mean().pow(.5)*np.sqrt(252)
up=resid.clip(lower=0).pow(2).rolling(40,min_periods=20).mean().pow(.5)*np.sqrt(252)
asym=((down+1e-8)/(up+1e-8)).clip(0.5,2.0)
# Contrarian residual trend, emphasizing assets with downside-dominant recent risk
sig=(-res60/(down+1e-8)*asym).replace([np.inf,-np.inf],np.nan)
sig=sig.sub(sig.median(axis=1),axis=0).shift(1)
for h in [5,10,20,40]:
    fwd=c.shift(-h)/c-1; rows=[]
    for dt in c.index:
        z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
    q=pd.DataFrame(rows,columns=['date','ic','n']); x=q.ic
    print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/len(U),4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
    if h==20:
        for name,a,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-12-31')]:
            y=q[(q.date>=a)&(q.date<=b)]; print('REG',name,len(y),round(y.ic.mean(),6),round(y.ic.mean()/y.ic.std(ddof=1),6) if len(y)>1 else np.nan,round((y.ic>0).mean(),4) if len(y) else np.nan)
        print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20310724_downside_asymmetry_reversal_signal.csv',index=False)
print('UNIVERSE',len(D),'DATES',len(c),'SIGNAL_ROWS',len(out))
