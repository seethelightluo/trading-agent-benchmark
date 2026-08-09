import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); px[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); v=r.rolling(20,min_periods=15).std(); cs=r.median(axis=1)
f=-(r.sub(cs,axis=0)).div(v); y=r.shift(-1); f=f.replace([np.inf,-np.inf],np.nan)
res=[]; art=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: res.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
ser=pd.Series({d:i for d,i,n in res}); print('dates',len(ser),'avg_n',np.mean([n for d,i,n in res]),'coverage',sum(n for d,i,n in res)/(len(res)*15));print('IC',ser.mean(),'ICIR',ser.mean()/ser.std(),'hit',(ser>0).mean(),'turnover',f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=ser.loc[a:b];print(a,b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20261217_residual_reversal1_signal.csv',index=False)
