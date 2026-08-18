import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): P[a]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2034-03-15']; R=P.pct_change()
d=R.where(R<0,0).rolling(40,min_periods=25).apply(lambda x: np.sqrt(np.mean(x*x))*np.sqrt(252),raw=True)
v=R.rolling(40,min_periods=25).std()*np.sqrt(252)
den=.7*d+.3*v
f20=(-P.pct_change(20)/den).shift(1); f30=(-P.pct_change(30)/den).shift(1)
F=.5*f20+.5*f30
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1
 z=pd.concat([F.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): rows.append((dt,len(z),ic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(P.columns))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(k); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
for h in [5,20]:
 yy=P.shift(-h)/P-1; rr=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],yy.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,round(np.nanmean(rr),6),len(rr))
print('coverage',round(F.notna().sum(axis=1).mean()/len(P.columns),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_1_20340316_blended_downside_contrarian_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_1_20340316_blended_downside_contrarian_signal.csv')
