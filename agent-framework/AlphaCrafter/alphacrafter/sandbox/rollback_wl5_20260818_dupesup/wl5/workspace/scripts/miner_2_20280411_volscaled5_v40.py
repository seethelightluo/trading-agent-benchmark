import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
base='../persistent/stock_data'; syms=sorted(x[:-4] for x in os.listdir(base) if x.endswith('.csv'))
px={}
for s in syms:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Candidate: five-day reversal weighted by short-vs-long volatility shock; longer denominator reduces noise
v5=r.rolling(5,min_periods=5).std(); v40=r.rolling(40,min_periods=40).std()
f=-(P.pct_change(5))*((v5/(v40+1e-12)).clip(0,5))
fr=P.shift(-10)/P.shift(-1)-1; rows=[]
for dt in P.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('candidate volscaled5_v40 universe',len(syms),'dates',len(o),'meanN',round(o.n.mean(),3),'coverage',round(o.n.mean()/len(syms),4))
print('IC',round(o.ic.mean(),6),'ICIR',round(o.ic.mean()/o.ic.std(ddof=1),6),'hit',round((o.ic>0).mean(),4))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
 q=o.loc[a:b].ic
 if len(q): print('regime',a,b,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for k in [60,120,252]:
 q=o.tail(k).ic; print('recent',k,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
rank=f.rank(axis=1,pct=True); rr=[]
for i in range(1,len(rank)):
 z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(z)>=8: rr.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('turnover_proxy',round(float(np.nanmean(rr)),6))
# artifact for post-gate reproducibility
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_2_20280411_volscaled5_v40_signal.csv',index=False)
print('artifact','scripts/miner_2_20280411_volscaled5_v40_signal.csv')
