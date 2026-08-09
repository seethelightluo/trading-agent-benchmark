import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; z={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); z[s]=d[['open','close','high','low']].astype(float)
P=pd.concat(z,axis=1).sort_index(); o=P.xs('open',axis=1,level=1); c=P.xs('close',axis=1,level=1); hi=P.xs('high',axis=1,level=1); lo=P.xs('low',axis=1,level=1)
rng=(hi-lo).replace(0,np.nan); # close location, reversed to predict next return
clv=2*(c-lo)/rng-1
f=-clv.rolling(3,min_periods=3).mean(); f=f.sub(f.median(axis=1),axis=0); fr=c.pct_change().shift(-1)
rows=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(q)>=8: rows.append((dt,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
s=pd.Series({x[0]:x[1] for x in rows}); print('dates',len(s),'avg_n',np.mean([x[2] for x in rows]),'coverage',sum(x[2] for x in rows)/(len(rows)*15)); print('daily IC %.8f ICIR %.8f hit %.5f turnover %.8f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=s.loc[a:b]; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20261217_clv3_reversal_signal.csv',index=False)
