import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2027-02-24']
r=P.pct_change(); # lagged 20d residual from cross-sectional median, normalized by cross-sectional MAD
m=r.rolling(20,min_periods=15).sum().shift(1)
med=m.median(axis=1); mad=(m.sub(med,axis=0).abs()).median(axis=1)
f=-(m.sub(med,axis=0)).div(mad.replace(0,np.nan),axis=0)
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; vals=[]; dates=[]; ns=[]
 for d in f.index:
  a=f.loc[d]; b=y.loc[d]
  z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(d); ns.append(len(z))
 v=np.array(vals); print('h',h,'dates',len(v),'avgN',round(np.mean(ns),2),'IC',round(np.mean(v),5),'ICIR',round(np.mean(v)/(np.std(v,ddof=1)+1e-12)*np.sqrt(len(v)),5),'hit',round(np.mean(v>0),4))
 for lo,hi in [('2024','2024'),('2025','2025'),('2026','2026'),('2027','2027')]:
  q=v[[str(d)[:4]==lo for d in dates]]; print(lo, len(q), round(np.mean(q),5) if len(q) else None)
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
# signal artifact for admission horizon 5
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_1_20270225_dispersion_reversal20.csv',index=False)
