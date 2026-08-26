import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=3000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
down=r.where(r<0).rolling(30,min_periods=15).std()
raw=(p.pct_change(20)/down).replace([np.inf,-np.inf],np.nan)
f=raw.sub(raw.median(axis=1),axis=0).shift(1)
rows=[]
for h in [1,3,5,10]:
    fr=p.pct_change(h).shift(-h); vals=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    a=pd.Series(vals).dropna(); rows.append((h,len(a),a.mean(),a.mean()/a.std(),(a>0).mean()))
x=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:x.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.Series(dict(x)); n=len(a); thirds=[a.iloc[:n//3],a.iloc[n//3:2*n//3],a.iloc[2*n//3:]]
print('dates',len(p),'usable',n,'avgN',f.notna().sum(axis=1).mean())
print('metrics',rows)
print('regimes',[round(q.mean(),6) for q in thirds])
print('turnover',raw.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean())
