import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-03-25'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close.loc[:end]
P=pd.DataFrame(px).ffill(); R=P.pct_change(); r5=P.pct_change(5); r20=P.pct_change(20); vol=R.rolling(20).std()*np.sqrt(252)
f=(r20/4-r5)/vol.replace(0,np.nan)
allres={}
for h in [1,5,10]:
 fr=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); m=z.ic.mean(); ir=m/z.ic.std(ddof=1)*np.sqrt(252)
 allres[h]=(len(z),z.n.mean(),m,ir,(z.ic>0).mean())
 print('h',h,'dates',len(z),'avgN',z.n.mean(),'IC',m,'ICIR',ir,'hit',(z.ic>0).mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-25')]:
 q=[]
 for dt in f.index:
  if not(a<=str(dt.date())<=b): continue
  x=f.loc[dt]; y=(P.shift(-1)/P-1).loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 q=np.array(q); print('regime',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
rows=[]
for dt in f.index:
 for s in U: rows.append((dt,s,f.loc[dt,s]))
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20270325_deceleration_reversal_signal.csv',index=False)
