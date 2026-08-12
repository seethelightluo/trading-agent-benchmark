import os, json
import numpy as np
import pandas as pd

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    p=os.path.join(base,s+'.csv')
    d=pd.read_csv(p)
    d['date']=pd.to_datetime(d['date'])
    d=d.sort_values('date').set_index('date')
    px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index()
R=P.pct_change()
# Risk-adjusted medium momentum: 20-session cumulative return divided by trailing 20-session realized volatility.
# Signal is observable at t; forward return is t+1, with no look-ahead.
sig=(P/P.shift(20)-1)/(R.rolling(20,min_periods=15).std()*np.sqrt(252))
fwd=P.shift(-1)/P-1
rows=[]; artifact=[]
for dt in P.index:
    x=sig.loc[dt]; y=fwd.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(z)))
        for s in z.index: artifact.append((dt,s,float(x[s])))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mu=q.ic.mean(); sd=q.ic.std(ddof=1); icir=mu/sd*np.sqrt(252) if sd else np.nan
# daily paper convention is mean IC / std IC (not annualized)
daily_icir=mu/sd if sd else np.nan
# signal turnover: mean fraction rank changes across consecutive dates, over common names
rank=sig.rank(axis=1,pct=True)
turn=rank.diff().abs().mean(axis=1).mean()
coverage=q.n.mean()/15
print(json.dumps({'dates':len(q),'date_start':str(q.index.min().date()),'date_end':str(q.index.max().date()),'avg_instruments':q.n.mean(),'daily_ic':mu,'daily_icir':daily_icir,'annualized_icir':icir,'hit_ratio':(q.ic>0).mean(),'coverage':coverage,'turnover':turn},indent=2))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2031-08-21')]:
 t=q.loc[a:b]
 if len(t): print(a,b,len(t),float(t.ic.mean()),float(t.ic.mean()/t.ic.std(ddof=1)),float((t.ic>0).mean()))
os.makedirs('scripts',exist_ok=True)
pd.DataFrame(artifact,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20310821_risk_adjusted_momentum_20d_signal.csv',index=False)
