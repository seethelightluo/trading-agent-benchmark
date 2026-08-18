import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); D[s]=x.set_index('date').close
P=pd.DataFrame(D).sort_index().loc[:'2030-07-24']
R=P.pct_change()
# 20-session momentum scaled by trailing 20d realized volatility; fully observable at t
F=R.rolling(20).sum()/R.rolling(20).std()
# use signals at t and forward returns t+1..t+h
for h in [1,5,10,20]:
 vals=[]; Ns=[]
 for i in range(len(P)-h):
  f=F.iloc[i]; fr=P.iloc[i+h]/P.iloc[i]-1
  z=pd.concat([f,fr],axis=1).dropna();
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); Ns.append(len(z))
 a=np.array(vals); recent=a[-260:]
 print(f'h={h} dates={len(a)} avgN={np.mean(Ns):.2f} IC={np.nanmean(a):.6f} ICIR={np.nanmean(a)/np.nanstd(a,ddof=1):.6f} hit={np.mean(a>0):.4f} recentIC={np.nanmean(recent):.6f} recentICIR={np.nanmean(recent)/np.nanstd(recent,ddof=1):.6f}')
print('assets',len(D),'dates',len(P),'coverage',F.notna().sum().sum()/(len(P)*len(D)))
# rank turnover proxy
r=F.rank(axis=1,pct=True); print('turnover',np.nanmean((r-r.shift(1)).abs().mean(axis=1)))
