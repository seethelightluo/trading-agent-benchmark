import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
# aligned panel, use close; signal at t lagged one session via shift(1)
px=pd.DataFrame({s:d[s].close for s in U}).sort_index(); ret=np.log(px/px.shift(1))
vol=ret.rolling(20,min_periods=15).std()
base=ret.rolling(20,min_periods=20).sum()/vol - ret.rolling(60,min_periods=50).sum()/ret.rolling(60,min_periods=50).std()
base=base.replace([np.inf,-np.inf],np.nan).shift(1)
fwd=np.log(px.shift(-10)/px)
for k in (3,10):
    sig=base.rolling(k,min_periods=k).mean()
    rows=[]; vals=[]
    for dt in px.index:
        x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
            if np.isfinite(ic): rows.append((dt,ic,len(z)))
    q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
    daily=q.ic; ic=daily.mean(); sd=daily.std(ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
    print('K',k,'dates',len(q),'meanN',q.n.mean(),'coverage',q.n.mean()/15,'IC',ic,'ICIR',icir,'hit',(daily>0).mean(),'turnover',sig.rank(axis=1).diff().abs().sum(axis=1).div(15*14).mean())
    for w in (120,252,504):
        z=daily.tail(w); print(' recent',w,z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252) if len(z)>2 else np.nan)
    # horizon decay
    for h in (1,5,10,20):
        fy=np.log(px.shift(-h)/px); a=[]
        for dt in px.index:
            z=pd.concat([sig.loc[dt],fy.loc[dt]],axis=1).dropna()
            if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
        a=pd.Series(a).dropna(); print(' h',h,'IC',a.mean(),'IR',a.mean()/a.std(ddof=1)*np.sqrt(252))
