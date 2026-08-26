import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-05-03'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); eq=S[:8]; market=r[eq].mean(axis=1); rows=[]
for i,dt in enumerate(r.index):
 if i<31: continue
 win=r.iloc[i-20:i]; down=win.loc[market.iloc[i-20:i]<0]
 if len(down)<5: continue
 # higher is better: less downside return per unit total volatility
 f=-(down.mean()/win.std(ddof=1)).replace([np.inf,-np.inf],np.nan)
 y10=px.shift(-10).loc[dt]/px.loc[dt]-1
 z=pd.concat([f,y10],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
A=pd.DataFrame(rows,columns=['date','n','ic'])
for label,q in [('all',A),('online',A[A.date>=pd.Timestamp('2026-07-16')]),('recent',A[A.date>=pd.Timestamp('2027-05-04')])]:
 print(label,'dates',len(q),'mean_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
print('period',px.index.min().date(),end.date(),'universe',len(S),'coverage',round(A.n.mean()/15,4))
