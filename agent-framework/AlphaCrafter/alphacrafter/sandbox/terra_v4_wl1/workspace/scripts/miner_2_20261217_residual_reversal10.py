import numpy as np, pandas as pd

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# 10-session volatility-scaled residual reversal; only completed bars returned by API
allx={}
for s in U:
    d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
    if len(d):
        x=d[['date','close']].copy(); x=x.drop_duplicates('date').set_index('date').sort_index()
        allx[s]=x.close.astype(float)
p=pd.DataFrame(allx).sort_index(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
# signal at t predicts t+1 close return
raw=r.rolling(10,min_periods=8).sum(); med=raw.median(axis=1)
f=-(raw.sub(med,axis=0)).div(vol)
f=f.replace([np.inf,-np.inf],np.nan); fr=r.shift(-1)
ics=[]; rows=[]
for dt in f.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); ics.append(ic); rows.append((dt,ic,len(z)))
ser=pd.Series(dict((x[0],x[1]) for x in rows)).sort_index()
print('dates',len(ser),'avg_n',np.mean([x[2] for x in rows]),'coverage',sum(x[2] for x in rows)/(len(rows)*len(U)))
print('daily IC %.8f ICIR %.8f hit %.5f turnover %.8f'%(ser.mean(),ser.mean()/ser.std(ddof=1), (ser>0).mean(), f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for h in [5,10]:
    rr=p.pct_change(h).shift(-h); q=[]
    for dt in f.index:
      z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
      if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    q=pd.Series(q); print('%dd IC %.8f ICIR %.8f'%(h,q.mean(),q.mean()/q.std(ddof=1)))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=ser.loc[a:b]; print(a,b,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
# artifact with date,symbol,signal for audit
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20261217_residual_reversal10_signal.csv',index=False)
