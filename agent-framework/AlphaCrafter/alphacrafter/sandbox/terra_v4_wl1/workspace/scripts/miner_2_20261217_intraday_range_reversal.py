import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
a={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 a[s]=d[['open','close','high','low']].astype(float)
# robust intraday reversal: prior intraday return, scaled by prior true range, cross-sectional demean
P=pd.concat(a,axis=1).sort_index(); o=P.xs('open',axis=1,level=1); c=P.xs('close',axis=1,level=1); hi=P.xs('high',axis=1,level=1); lo=P.xs('low',axis=1,level=1)
intra=c/o-1; tr=(hi-lo)/c.shift(1); f=-(intra/tr.clip(lower=1e-5)).clip(-5,5); f=f.sub(f.median(axis=1),axis=0); fr=c.pct_change().shift(-1)
ics=[]; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); rows.append((dt,ics[-1],len(z)))
ser=pd.Series(dict((x[0],x[1]) for x in rows)).sort_index(); print('dates',len(ser),'avg_n',np.mean([x[2] for x in rows]),'coverage',sum(x[2] for x in rows)/(len(rows)*15)); print('daily IC %.8f ICIR %.8f hit %.5f turnover %.8f'%(ser.mean(),ser.mean()/ser.std(ddof=1),(ser>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for a1,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=ser.loc[a1:b]; print(a1,b,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20261217_intraday_range_reversal_signal.csv',index=False)
print('artifact scripts/miner_2_20261217_intraday_range_reversal_signal.csv')
