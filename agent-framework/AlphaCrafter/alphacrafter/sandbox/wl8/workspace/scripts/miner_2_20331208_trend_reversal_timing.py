import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,4000)
 except Exception: pass
 if d is None:
  try:d=get_stock_daily_data(s,4000)
  except Exception: pass
 if d is not None and len(d)>150: raw[s]=d[['date','close']].set_index('date')['close'].astype(float)
idx=sorted(set.intersection(*[set(x.index) for x in raw.values()])); px=pd.DataFrame({s:raw[s].reindex(idx) for s in raw}).sort_index(); r=np.log(px).diff()
f=(r.rolling(60).sum()/60-r.rolling(10).sum()/10).shift(1).rolling(3).mean(); fr=np.log(px.shift(-10)/px)
rows=[]; ranks=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z))); ranks.append(f.loc[dt].rank(pct=True))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); sr=pd.DataFrame(ranks,index=q.index)
for lab,z in [('all',q),('r720',q.tail(720)),('r365',q.tail(365)),('r180',q.tail(180))]:
 m=z.ic.mean(); print(lab,len(z),round(z.n.mean(),2),round(m,6),round(m/z.ic.std(ddof=1),6),round((z.ic>0).mean(),4))
print('coverage',round(q.n.sum()/(len(q)*15),4),'turnover',round(float(sr.diff().abs().mean().mean()),6),'assets',len(raw),'period',q.index.min(),q.index.max())
for h in [1,5,20]:
 a=[]; ff=np.log(px.shift(-h)/px)
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,round(np.mean(a),6),len(a))
q.to_csv('scripts/miner_2_20331208_trend_reversal_timing_ic.csv'); f.to_csv('scripts/miner_2_20331208_trend_reversal_timing_signal.csv')
