import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,3000)
 except:pass
 if d is None:
  try:d=get_stock_daily_data(s,3000)
  except:pass
 if d is not None and len(d)>100:raw[s]=d[['date','close']].set_index('date')
idx=sorted(set.intersection(*[set(v.index) for v in raw.values()])); P=pd.DataFrame({s:raw[s].reindex(idx).close for s in raw}).sort_index(); R=P.pct_change()
# Residual reversal: fade 10-session return after removing equal-weight cross-asset movement, lagged one session.
res=R-R.mean(axis=1).values[:,None]
f=(-res.rolling(10).sum()).shift(1); fw=P.shift(-10)/P-1
rows=[]; ranks=[]
for dt in P.index:
 a,b=f.loc[dt],fw.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8:rows.append((dt,a[ok].corr(b[ok],method='spearman'),ok.sum()));ranks.append(a.rank(pct=True))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); sr=pd.DataFrame(ranks,index=q.index)
for lab,z in [('all',q),('r720',q.tail(720)),('r360',q.tail(360)),('r180',q.tail(180)),('r60',q.tail(60))]:
 print(lab,len(z),round(z.n.mean(),2),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1),6),round((z.ic>0).mean(),4))
print('coverage',round(sum(q.n)/(len(q)*15),4),'turnover',round(float(sr.diff().abs().mean().mean()),6),'assets',len(raw),'period',q.index.min(),q.index.max())
f.to_csv('scripts/miner_2_20310710_residual_reversal_signal.csv');q.to_csv('scripts/miner_2_20310710_residual_reversal_ic.csv')
