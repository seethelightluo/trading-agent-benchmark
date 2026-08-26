import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None: break
  except Exception: pass
 if d is not None and len(d)>250: raw[s]=d[['date','close']].set_index('date')['close'].astype(float)
idx=sorted(set.intersection(*[set(x.index) for x in raw.values()])); px=pd.DataFrame({s:raw[s].reindex(idx) for s in raw}).sort_index(); r=np.log(px).diff()
# Cross-asset dispersion gate: short reversal is strongest when cross-sectional
# 20d return dispersion is high, avoiding low-opportunity synchronized markets.
ret10=r.rolling(10).sum(); disp=ret10.std(axis=1).rolling(20).mean(); med=disp.rolling(120).median(); gate=(disp/med).clip(.5,2)
f=(-ret10.mul(gate,axis=0)).shift(1).rolling(3).mean(); fr=np.log(px.shift(-10)/px)
rows=[]; ranks=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z))); ranks.append(f.loc[dt].rank(pct=True))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); sr=pd.DataFrame(ranks,index=q.index)
for lab,z in [('all',q),('r720',q.tail(720)),('r365',q.tail(365)),('r180',q.tail(180))]:
 m=z.ic.mean(); print(lab,'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(m,6),'ICIR',round(m/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(q.n.sum()/(len(q)*len(raw)),4),'turnover',round(float(sr.diff().abs().mean().mean()),6),'assets',len(raw),'period',q.index.min(),q.index.max())
for h in [1,5,20]:
 a=[]; ff=np.log(px.shift(-h)/px)
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,round(np.mean(a),6),len(a))
q.to_csv('scripts/miner_2_20340119_dispersion_reversal_ic.csv'); f.to_csv('scripts/miner_2_20340119_dispersion_reversal_signal.csv')
