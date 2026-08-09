import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv'); P={}
for f in files:
 d=pd.read_csv(f); P[os.path.basename(f)[:-4]]=d.set_index('date')['close']
P=pd.DataFrame(P).sort_index(); R=np.log(P).diff()
# volume-confirmed medium momentum: 10d return multiplied by normalized 20d volume surprise
V={}
for f in files:
 d=pd.read_csv(f); V[os.path.basename(f)[:-4]]=d.set_index('date')['volume']
V=pd.DataFrame(V).reindex(P.index); vz=(V/V.rolling(40,min_periods=20).median()-1).clip(-2,2)
sig=np.log(P/P.shift(10))* (1+0.5*vz)
for h in [1,5,10]:
 F=np.log(P.shift(-h)/P); ic=[]; ns=[]; ds=[]
 for i in range(1,len(P)-h):
  ok=sig.iloc[i].notna()&F.iloc[i].notna()
  if ok.sum()>=8: ic.append(spearmanr(sig.iloc[i][ok],F.iloc[i][ok]).statistic);ns.append(ok.sum());ds.append(P.index[i])
 z=np.array(ic); print('H',h,'dates',len(z),'meanN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',np.mean(z),'ICIR',np.mean(z)/(np.std(z,ddof=1)+1e-12),'hit',np.mean(z>0))
for a,b in [('2020','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=np.array([x for x,d in zip(ic,ds) if d>=a+'-01-01' and d<=b+'-12-31']); print(a,b,len(q),np.mean(q) if len(q) else np.nan)
print('assets',len(P.columns))
