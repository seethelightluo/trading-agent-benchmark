import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index(); P[a]=d.close
P=pd.DataFrame(P).loc[:'2033-09-28']; R=P.pct_change(); down=R.clip(upper=0).rolling(30,min_periods=20).std(); F=(-P.pct_change(10)/(down*np.sqrt(10))).shift(1)
rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],P.shift(-10).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',r.n.mean(),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
for k in [120,260,520,780]:
 q=s.tail(k);print('recent',k,q.mean(),q.mean()/q.std())
print('coverage',F.notna().sum(axis=1).mean()/15,'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
for h in [1,5,10,20,30]:
 z=[]
 for dt in F.index:
  x=pd.concat([F.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(x)>=8:z.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(z),len(z))
os.makedirs('scripts/artifacts',exist_ok=True);r.to_csv('scripts/artifacts/miner_1_20330929_downside_risk_reversal_ic.csv',index=False);F.to_csv('scripts/artifacts/miner_1_20330929_downside_risk_reversal_signal.csv')
