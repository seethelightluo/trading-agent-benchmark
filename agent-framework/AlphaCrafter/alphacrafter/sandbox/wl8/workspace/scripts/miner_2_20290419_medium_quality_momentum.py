import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    try:
        d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
        frames[s]=d['close'].astype(float)
    except Exception as e: print('missing',s)
p=pd.DataFrame(frames).sort_index(); r=np.log(p).diff()
f=r.rolling(20).sum()/r.rolling(20).std().replace(0,np.nan)
for h in [1,3,5,10]:
    fw=np.log(p.shift(-h)/p); vals=[]; dates=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
    a=pd.Series(vals,index=dates).dropna()
    print('H',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'dates',len(a),'avgN',np.mean(ns),'coverage',len(a)/len(f.index))
    for label,lo,hi in [('2026','2026-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('recent','2028-10-01','2029-04-19')]:
        q=a[(a.index>=lo)&(a.index<=hi)]
        print(label,round(q.mean(),5),round(q.mean()/q.std(ddof=1),5),len(q))
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean(),'valid',f.notna().mean().mean(),'rows',len(p))
