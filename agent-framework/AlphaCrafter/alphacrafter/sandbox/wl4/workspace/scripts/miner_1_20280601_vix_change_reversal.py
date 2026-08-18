import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); P=P[P.index<=pd.Timestamp('2028-05-31')]; R=P.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
# VIX change surprise relative to trailing 60d volatility; bounded multiplier avoids outliers.
shock=v.pct_change(5)/v.pct_change().rolling(60).std().replace(0,np.nan)
mult=(1+0.25*shock.clip(-2,2)).clip(.5,1.5)
F=-R.rolling(5).sum().mul(mult,axis=0)
rows=[]; turns=[]
for i in range(len(P)-1):
 z=pd.concat([F.iloc[i],(P.shift(-1).iloc[i]/P.iloc[i]-1)],axis=1).dropna()
 if len(z)>=8: rows.append((P.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 if i>0: turns.append((F.iloc[i-1].rank(pct=True)-F.iloc[i].rank(pct=True)).abs().mean())
a=pd.DataFrame(rows,columns=['date','ic','n']); ic=a.ic; recent=ic.tail(250)
print('dates',len(a),'avgN',round(a.n.mean(),2),'minN',a.n.min(),'coverage',round(a.n.sum()/(len(a)*15),4))
print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'turn',round(np.nanmean(turns),6),'recentIC',round(recent.mean(),6),'recentIR',round(recent.mean()/recent.std(ddof=1),6))
for h in [5,10,20]:
 yy=P.shift(-h)/P-1; rr=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],yy.iloc[i]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'IC',round(np.nanmean(rr),6),'ICIR',round(np.nanmean(rr)/np.nanstd(rr,ddof=1),6),'dates',len(rr))
