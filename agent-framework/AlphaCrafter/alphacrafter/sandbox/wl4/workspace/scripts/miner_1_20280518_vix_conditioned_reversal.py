import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); P=P[P.index<=pd.Timestamp('2028-05-18')]; ret=P.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']; v=v.reindex(P.index).ffill(); mult=v/v.rolling(60).median(); gate=(v>v.rolling(120).quantile(.7)).astype(float)
basef=-ret.rolling(5).sum(); variants={'vix_level':basef.mul(mult,axis=0),'vix_shock':basef.mul(mult.clip(.5,2),axis=0),'vix_high_gate':basef.mul(gate,axis=0),'vix_low_gate':basef.mul(1-gate,axis=0)}
print('P',P.shape,'VIX valid',v.notna().sum())
for name,F in variants.items():
 rows=[]; turnovers=[]
 for i in range(len(P)-1):
  x=F.iloc[i]; y=P.iloc[i+1]/P.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: rows.append((P.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  if i>0: turnovers.append((F.iloc[i-1].rank(pct=True)-x.rank(pct=True)).abs().mean())
 a=pd.DataFrame(rows,columns=['date','ic','n']); ic=a.ic; recent=ic.tail(250)
 print(name,'dates',len(a),'avgN',round(a.n.mean(),2),'coverage',round(a.n.sum()/(len(a)*15),4),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'turn',round(np.nanmean(turnovers),6),'recentIC',round(recent.mean(),6),'recentIR',round(recent.mean()/recent.std(ddof=1),6))
 for h in [5,10,20]:
  yy=P.shift(-h)/P-1; rr=[]
  for i in range(len(P)-h):
   z=pd.concat([F.iloc[i],yy.iloc[i]],axis=1).dropna()
   if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  print(' ',h,round(np.nanmean(rr),6),round(np.nanmean(rr)/np.nanstd(rr,ddof=1),6),len(rr))
