import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in A:
    q=Path('../persistent/stock_data')/(s+'.csv')
    d=pd.read_csv(q,parse_dates=['date']).set_index('date')['close']
    px[s]=d.sort_index()
p=pd.concat(px,axis=1).sort_index().loc[:'2032-12-22']
r=p.pct_change(); v5=r.rolling(5,min_periods=4).std(); v30=r.rolling(30,min_periods=20).std()
# reversal is stronger after an abnormal recent volatility shock
sig=-(p.pct_change(5))*v5.div(v30).replace(0,np.nan)
for h in [5,10,20,40]:
    f=p.shift(-h).div(p)-1; vals=[]; ns=[]; dates=[]
    for i in range(len(p)-h):
        z=pd.concat([sig.iloc[i],f.iloc[i]],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(p.index[i])
    x=np.array(vals); print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4))
    print('regimes',pd.Series(x,index=pd.DatetimeIndex(dates)).groupby(lambda z:z.year).mean().round(5).to_dict())
u=[]
for i in range(1,len(sig)):
    z=pd.concat([sig.iloc[i-1].rank(pct=True),sig.iloc[i].rank(pct=True)],axis=1).dropna()
    if len(z): u.append(np.mean(abs(z.iloc[:,1]-z.iloc[:,0])))
print('turnover',round(float(np.mean(u)),6))
