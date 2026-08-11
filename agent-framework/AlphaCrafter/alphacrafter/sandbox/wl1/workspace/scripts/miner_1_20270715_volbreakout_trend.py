import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=4000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index()
        D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
r=p.pct_change()
# Candidate: volatility-breakout trend: medium momentum, scaled by recent volatility,
# with a continuous confirmation from short-vs-medium trend. All inputs lagged one day.
ret20=p.shift(1)/p.shift(21)-1
ret5=p.shift(1)/p.shift(6)-1
vol20=r.rolling(20).std().shift(1)*np.sqrt(252)
# sign confirmation emphasizes trends that persist rather than one-day reversals
f=(ret20/vol20.replace(0,np.nan))*(1+0.5*np.tanh(ret5/vol20.replace(0,np.nan)))
# forward close-to-close returns
out=[]
for h in [5,10,20]:
    fr=p.shift(-h)/p-1
    vals=[]
    for dt in f.index:
        a=f.loc[dt]; b=fr.loc[dt]
        z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    q=pd.Series(vals).dropna()
    out.append((h,len(q),q.mean(),q.mean()/q.std(ddof=1), (q>0).mean()))
# coverage and rank turnover
valid=f.notna().sum(axis=1); cov=valid.div(len(U)).mean()
ranks=f.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).mean()
print('dates',len(p),'instruments',len(D),'valid_dates',len(q),'avg_inst',valid.mean(),'coverage',cov,'turnover',turnover)
for x in out: print('horizon=%d dates=%d IC=%.9f ICIR=%.9f hit=%.4f'%x)
# recent/regime split
for name,mask in [('2020-2024',f.index<'2025-01-01'),('2025-2026', (f.index>='2025-01-01')&(f.index<'2027-01-01')),('2027',f.index>='2027-01-01')]:
    fr=p.shift(-20)/p-1; vals=[]
    for dt in f.index[mask]:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    q=pd.Series(vals).dropna(); print(name,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
