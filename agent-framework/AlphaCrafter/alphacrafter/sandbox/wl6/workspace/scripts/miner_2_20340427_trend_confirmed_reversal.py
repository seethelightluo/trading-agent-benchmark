import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x)>100: D[s]=x.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index().ffill()
R=P.pct_change()
# trend-confirmed short-term reversal: reverse recent move, with a modest boost when
# the 60d trend agrees with the reversal (buy dips in uptrends, sell rallies in downtrends)
r10=P/P.shift(10)-1; r60=P/P.shift(60)-1
trend=np.sign(r60)
factor= -r10*(1+0.5*trend)
# lag factor to ensure only completed close is used
F=factor.shift(1)
print('data',P.index.min().date(),P.index.max().date(),'dates',len(P),'assets',len(D))
for h in [5,10,20,40]:
    fr=P.shift(-h)/P-1
    vals=[]; dates=[]; nms=[]
    for dt in F.index:
        a=F.loc[dt]; b=fr.loc[dt]
        z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); nms.append(len(z))
    q=pd.Series(vals,index=dates).dropna(); print('H',h,'dates',len(q),'avgN',np.mean(nms),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
# rank turnover proxy
r=F.rank(axis=1,pct=True); turn=(r.diff().abs().mean(axis=1)).mean(); cov=F.notna().sum(axis=1).mean()/len(U)
print('turnover_proxy',turn,'coverage',cov)
# decade/regime summaries at 10d
fr=P.shift(-10)/P-1; z=[]
for dt in F.index:
 a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:z.append((dt,a.iloc[:,0].corr(a.iloc[:,1],method='spearman')))
q=pd.Series(dict(z)).dropna(); print('annual',q.groupby(q.index.year).agg(['mean','count']).to_string())
