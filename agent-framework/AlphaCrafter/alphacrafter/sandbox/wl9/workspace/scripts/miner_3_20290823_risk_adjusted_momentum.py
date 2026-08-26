import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in ASSETS:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.dropna().drop_duplicates('date').set_index('date').sort_index(); px[s]=x.close
p=pd.DataFrame(px).sort_index(); ret=p.pct_change()
f=(p/p.shift(60)-1)/(ret.rolling(20,min_periods=15).std()*np.sqrt(20))
for horizon in [5,10,20]:
    fr=p.shift(-horizon)/p-1; obs=[]; cover=[]; pairs=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
            if pd.notna(c): obs.append(c); cover.append(len(z)/len(ASSETS)); pairs.append((dt,c))
    ic=pd.Series(obs); print('H',horizon,'dates',len(ic),'meanN',np.mean(np.array(cover)*len(ASSETS)),'coverage',np.mean(cover),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0))
    ss=pd.Series(dict(pairs))
    for a,b in [('2020','2023'),('2024','2026'),('2027','2029')]:
        q=ss[(ss.index>=a)&(ss.index<=b+'-12-31')]; print('REG',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=f.dropna(how='all').tail(1).T.rename(columns={f.dropna(how='all').index[-1]:'signal'}); out.to_csv('scripts/miner_3_20290823_risk_adjusted_momentum_signal.csv')
