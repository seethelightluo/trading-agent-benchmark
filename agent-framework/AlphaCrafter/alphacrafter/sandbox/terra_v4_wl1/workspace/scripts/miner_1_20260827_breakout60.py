import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date'])
    d=d[d.date<=pd.Timestamp('2026-08-26')].sort_values('date').set_index('date')
    px[s]=d.close.astype(float)
P=pd.DataFrame(px).sort_index(); f=P/P.shift(1).rolling(60,min_periods=45).max()-1
for h in [1,5,10]:
    vals=[]
    for dt in P.index:
        x=f.loc[dt]; y=P.shift(-h).loc[dt]/P.loc[dt]-1
        z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
    a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
    print('h',h,'dates',len(a),'avgN',a.n.mean(),'IC %.5f ICIR %.5f hit %.4f cov %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean(),f.notna().stack().mean()))
    for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
        q=a.loc[lo:hi,'ic']; print(lo+'-'+hi,'n',len(q),'mean %.5f icir %.5f'%(q.mean(),q.mean()/q.std(ddof=1)))
r=f.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
f.to_csv('scripts/miner_1_20260827_breakout60_signal.csv')
