import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);D[s]=d.sort_values('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index();r=p.pct_change(); mom=p/p.shift(60)-1; v=r.rolling(60).std(); f=mom/v.replace(0,np.nan)
for h in [10,20]:
 z=[];ns=[]
 for i in range(len(p)-h):
  a=f.iloc[i];b=p.iloc[i+h]/p.iloc[i]-1;ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(spearmanr(a[ok],b[ok]).statistic);ns.append(ok.sum())
 z=np.array(z);print('h',h,'dates',len(z),'avg_n',np.mean(ns),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1),'hit',np.mean(z>0))
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
PY
python - <<'PY'
p='scripts/miner_1_20300613_lowvol_momentum.py';s=open(p).read().replace('\nPY\npython scripts/miner_1_20300613_lowvol_momentum.py','\n');open(p,'w').write(s)
PY
python scripts/miner_1_20300613_lowvol_momentum.py