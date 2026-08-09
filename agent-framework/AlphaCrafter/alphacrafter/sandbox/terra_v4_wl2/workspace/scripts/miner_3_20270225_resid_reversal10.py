import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; P={}
for s in U:
 d=pd.read_csv(os.path.join(b,s+'.csv')); d['date']=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(P); R=P.pct_change();
# medium-horizon residual reversal: 10d return relative to cross-sectional median, smoothed by 3d recent return
r10=P.pct_change(10); r3=P.pct_change(3); fac=-(r10-r10.median(axis=1,skipna=True).values[:,None]) - .35*(r3-r3.median(axis=1,skipna=True).values[:,None])
fwd=P.shift(-5)/P-1; rows=[]; sig=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z))); sig.append((dt,*fac.loc[dt,U]))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(a),'avgN',a.n.mean(),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(),'hit',(a.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 q=a.loc[lo:hi]; print(lo,len(q),q.ic.mean() if len(q) else None,q.ic.mean()/q.ic.std() if len(q)>1 else None)
print('turn',pd.DataFrame(sig,columns=['date']+U).set_index('date').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10]:
 fw=P.shift(-h)/P-1; rr=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,len(rr),np.mean(rr),np.mean(rr)/np.std(rr))
pd.DataFrame(sig,columns=['date']+U).to_csv('../persistent/factor_signals_miner_3_20270225_resid_reversal10.csv',index=False)
