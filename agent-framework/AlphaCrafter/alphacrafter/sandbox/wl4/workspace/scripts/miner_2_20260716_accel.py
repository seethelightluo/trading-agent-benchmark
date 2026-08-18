import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv';
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(f); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close
P=pd.DataFrame(D).sort_index(); R=P.pct_change()
# Trend acceleration: intermediate 20d return relative to slower 60d return, isolating improving trend
F=(P/P.shift(20)-1)-(P/P.shift(60)-1)
for h in [1,5,10]:
 out=[]; ns=[]
 for i in range(65,len(P)-h):
  z=pd.concat([F.iloc[i],(P.iloc[i+h]/P.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(out); print(h,'obs',len(x),'dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'recent250',x[-250:].mean())
print('coverage',F.notna().mean().mean(),'turn',F.rank(pct=True).diff().abs().mean().mean())
