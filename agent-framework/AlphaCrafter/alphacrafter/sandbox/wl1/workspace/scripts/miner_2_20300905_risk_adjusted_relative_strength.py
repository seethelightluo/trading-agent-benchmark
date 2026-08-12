import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-09-04'); base=Path('../persistent/stock_data')
D={}
for s in U:
 x=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 D[s]=x.close.loc[:cut]
idx=sorted(set().union(*[set(v.index) for v in D.values()]))
p=pd.DataFrame({s:v.reindex(idx) for s,v in D.items()}).ffill(); r=p.pct_change()
# Risk-adjusted relative strength: medium-term return relative to cross-section, penalized by downside risk.
ret=p/p.shift(40)-1
rel=ret.sub(ret.median(axis=1),axis=0)
down=r.clip(upper=0).pow(2).rolling(60,min_periods=40).mean().pow(.5)
raw=rel/(down*np.sqrt(60)+1e-8)
# 10-day signal averaging lowers rebalance noise; lag ensures completed data.
f=raw.rolling(10,min_periods=5).mean().shift(1)
for h in [1,5,10,20]:
 ic=[]; ns=[]; dates=[]
 fr=p.shift(-h)/p-1
 for dt in p.index:
  a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 x=pd.Series(ic); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
valid=f.notna().sum(axis=1); print('coverage',round((valid/15).mean(),4),'turnover',round(f.diff().abs().mean().mean(),6))
for label,lo in [('2020-2025','2020-01-01'),('2026+','2026-01-01'),('2029+','2029-01-01'),('2030YTD','2030-01-01')]:
 z=[];fr=p.shift(-1)/p-1
 for dt in p.index[p.index>=lo]:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=pd.Series(z);print(label,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
out=pd.DataFrame(f);out.to_csv('scripts/miner_2_20300905_risk_adjusted_relative_strength_signal.csv')
