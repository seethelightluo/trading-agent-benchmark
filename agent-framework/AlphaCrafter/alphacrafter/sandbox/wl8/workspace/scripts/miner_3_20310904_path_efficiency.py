import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,3000)
 except:pass
 if d is None:
  try:d=get_stock_daily_data(s,3000)
  except:pass
 if d is not None and len(d)>100: raw[s]=d[['date','close']].set_index('date')
idx=sorted(set.intersection(*[set(x.index) for x in raw.values()]))
P=pd.DataFrame({s:raw[s].reindex(idx).close for s in raw}).sort_index(); r=P.pct_change()
# Path-efficiency momentum: directional 20d return divided by total absolute daily movement.
# Lagged one day; rewards persistent trends and suppresses noisy rallies.
ret=P.pct_change(20); path=r.abs().rolling(20).sum(); f=(ret/path).shift(1)
rows=[]; ranks=[]
for dt in P.index:
 a=f.loc[dt]; b=(P.shift(-10)/P-1).loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok],method='spearman'),ok.sum())); ranks.append(a.rank(pct=True))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); sr=pd.DataFrame(ranks,index=q.index)
for lab,z in [('all',q),('r365',q.tail(365)),('r180',q.tail(180)),('r60',q.tail(60))]:
 m=z.ic.mean(); sd=z.ic.std(ddof=1); print(lab,len(z),round(z.n.mean(),2),round(m,6),round(m/sd,6),round((z.ic>0).mean(),4))
print('coverage',round(q.n.mean()/15,4),'turnover',round(float(sr.diff().abs().mean().mean()),6),'assets',len(raw),'period',q.index.min(),q.index.max())
f.to_csv('scripts/miner_3_20310904_path_efficiency_signal.csv');q.to_csv('scripts/miner_3_20310904_path_efficiency_ic.csv')
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; rr=[]
 for dt in P.index:
  a,b=f.loc[dt],fw.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8:rr.append(a[ok].corr(b[ok],method='spearman'))
 print('decay',h,len(rr),round(np.nanmean(rr),6))
