import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}).sort_index().loc[:'2035-08-15']
r=P.pct_change(); down=r.where(r<0,0.0)
dd=P/P.rolling(60,min_periods=40).max()-1
dr=down.rolling(20,min_periods=15).std()
F=(-dd/dr).replace([np.inf,-np.inf],np.nan).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8:
  z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(z): rows.append((dt,len(q),z))
o=pd.DataFrame(rows,columns=['date','n','ic']); s=o.ic
print('period',o.date.min().date(),o.date.max().date(),'dates',len(o),'avgN',round(o.n.mean(),2),'assets',len(A))
print('full',round(s.mean(),6),round(s.mean()/s.std(ddof=1),6),round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4))
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; rr=[]
 for dt in F.index:
  q=pd.concat([F.loc[dt],yy.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): rr.append(z)
 print('decay',h,round(np.mean(rr),6),round(np.mean(rr)/np.std(rr,ddof=1),6))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True); o.to_csv('scripts/artifacts/miner_2_20350816_drawdown_downside_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_2_20350816_drawdown_downside_reversal_signal.csv',index_label='date')