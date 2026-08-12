import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Shock-rebound: recent 3-session reversal, scaled by trailing volatility, with a trend filter
# signal at t uses closes through t; forward returns begin t+1.
px={}
for s in U:
    d=get_stock_daily_data(s, days=3000)
    if d is None or len(d)<100: d=get_index_daily_data(s, days=3000)
    if d is not None and len(d):
        x=d.copy(); x['date']=pd.to_datetime(x['date']); px[s]=x.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill()
r=P.pct_change()
# reversal of short shock, volatility scaled; damp reversal when long trend is strongly negative
vol=r.rolling(20,min_periods=15).std()
short=r.rolling(3,min_periods=3).sum()
trend=r.rolling(20,min_periods=15).sum()
F=(-short/vol)*(1+0.25*trend.clip(lower=0))
# cross-sectional IC by date, forward k-day simple return
for k in [1,3,5,10]:
    fr=P.shift(-k)/P-1
    vals=[]; ns=[]
    for dt in F.index:
        z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
    a=np.asarray(vals); print('H',k,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
# turnover rank proxy
ranks=F.rank(axis=1,pct=True); turn=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna().mean()
print('rows',len(P),'assets',len(P.columns),'coverage',round(F.notna().sum(axis=1).mean()/15,4),'turnover',round(turn,6),'period',P.index.min(),P.index.max())
# regime split by median cross-sectional market return
market=r.mean(axis=1)
fr=P.shift(-1)/P-1; allv=[]; up=[]; down=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); allv.append(q)
  (up if market.loc[dt]>=0 else down).append(q)
print('regime up',len(up),round(np.nanmean(up),6),'down',len(down),round(np.nanmean(down),6))
