import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Risk-adjusted medium-term momentum: 60d return divided by 20d realized vol,
# lagged one session. Cross-sectional rank is not used in calculation.
series={}
for s in U:
    d=get_stock_daily_data(s, days=4200)
    if d is not None and len(d)>300:
        x=d[['date','close']].copy().dropna().drop_duplicates('date').set_index('date')['close'].astype(float)
        series[s]=x
p=pd.DataFrame(series).sort_index()
r=np.log(p).diff()
vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
raw=p.pct_change(60)/vol
f=(-raw).shift(1)
rows=[]
for h in [5,10,20,40,60]:
    fr=p.shift(-h)/p-1
    vals=[]
    for dt in f.index:
        a=f.loc[dt]; b=fr.loc[dt]
        z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    q=pd.Series(vals).dropna()
    print('H',h,'dates',len(q),'avgN',round(len(U)*0+0,2),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
# metrics at 20d, plus regime summaries
h=20; fr=p.shift(-h)/p-1; vals=[]; dates=[]; ns=[]
for dt in f.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
        vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
q=pd.Series(vals,index=pd.to_datetime(dates)); print('DETAIL valid_dates',len(q),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/len(U),'turnover_proxy',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2031-11-12')]:
 z=q.loc[a:b]; print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20311113_risk_adjusted_momentum_signal.csv',index=False)
