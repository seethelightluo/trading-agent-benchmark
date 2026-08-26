import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change()
# Candidate: short-term reversal gated by broad market stress/trend; score is negative 5d return, amplified when breadth is weak.
r5=p.pct_change(5); r20=p.pct_change(20)
breadth=(r20>0).sum(axis=1)/r20.notna().sum(axis=1)
stress=(1-breadth).clip(0,1)
f=(-r5).mul(0.5+stress,axis=0)
rows=[]
for h in [5,10,20]:
  fr=f.shift(1); fw=p.shift(-h)/p-1
  vals=[]; dates=[]; ns=[]
  for dt in p.index:
    x=fr.loc[dt]; y=fw.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
      ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
      if np.isfinite(ic): vals.append(ic); dates.append(dt); ns.append(len(z))
  a=np.array(vals); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/(np.std(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4),'turnover',round(np.mean((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)).dropna()),4))
# save 10d signal artifact
out=pd.DataFrame(f,columns=U); out.index.name='date'; out.to_csv('scripts/miner_3_20300110_breadth_conditioned_reversal_10d_signal.csv')
print('range',p.index.min(),p.index.max())
