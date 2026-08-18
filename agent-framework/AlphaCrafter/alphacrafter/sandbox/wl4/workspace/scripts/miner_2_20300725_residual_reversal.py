import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p);x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close
P=pd.DataFrame(D).sort_index().loc[:'2030-07-24']; R=P.pct_change(); mom=R.rolling(5).sum(); F=mom.sub(mom.mean(axis=1),axis=0)*-1
for h in [1,5,10,20]:
 a=[];ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],(P.iloc[i+h]/P.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);q=a[-260:]
 print(f'h={h} dates={len(a)} avgN={np.mean(ns):.2f} IC={np.mean(a):.6f} ICIR={np.mean(a)/np.std(a,ddof=1):.6f} hit={np.mean(a>0):.4f} recentIC={np.mean(q):.6f} recentICIR={np.mean(q)/np.std(q,ddof=1):.6f}')
print('assets',len(D),'dates',len(P),'coverage',F.notna().sum().sum()/(len(P)*len(D)))
rr=F.rank(axis=1,pct=True);print('turnover',np.nanmean((rr-rr.shift()).abs().mean(axis=1)))
