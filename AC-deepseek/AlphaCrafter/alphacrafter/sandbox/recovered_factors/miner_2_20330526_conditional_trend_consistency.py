import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv'); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Conditional trend-consistency: medium return is rewarded only when short and long returns agree.
r5=p.pct_change(5); r20=p.pct_change(20); r60=p.pct_change(60)
cons=(np.sign(r5)==np.sign(r20)).astype(float)*(np.sign(r20)==np.sign(r60)).astype(float)
# retain a small reversal component when consistency is absent, with one-day lag
sig=(r20*(0.35+0.65*cons)).shift(1)
print('range',p.index.min(),p.index.max(),'assets',len(assets))
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]; ns=[]; dates=[]
 for d in p.index:
  z=pd.concat([sig.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(d)
 s=np.asarray(vals); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),3))
rank=sig.rank(axis=1,pct=True); print('coverage',sig.notna().sum().mean()/15,'turnover10',((rank-rank.shift(10)).abs().mean(axis=1)).mean(),'mean_valid',sig.notna().sum(axis=1).mean())
fw=p.shift(-10)/p-1
for lo,hi in [(2024,2027),(2028,2030),(2031,2033)]:
 x=[]
 for d in p.index[(p.index.year>=lo)&(p.index.year<=hi)]:
  z=pd.concat([sig.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('REGIME',lo,hi,len(x),round(np.mean(x),6),round(np.mean(x)/np.std(x),6) if len(x)>1 else np.nan)
print('candidate cells',int(sig.notna().sum().sum()))
