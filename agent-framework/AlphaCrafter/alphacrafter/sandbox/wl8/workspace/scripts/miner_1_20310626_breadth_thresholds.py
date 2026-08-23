import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,days=3000)
 except:pass
 if d is None:
  try:d=get_stock_daily_data(s,days=3000)
  except:pass
 if d is not None and len(d)>100:raw[s]=d[['date','close']].drop_duplicates('date').set_index('date')['close']
idx=sorted(set.intersection(*[set(x.index) for x in raw.values()]));P=pd.DataFrame({s:raw[s].reindex(idx) for s in raw}).sort_index(); mom=P.pct_change(20).shift(1); br=(mom>0).mean(axis=1).shift(1); fw=P.shift(-10)/P-1
for th in [.33,.40,.50]:
 rows=[]
 for dt in P.index:
  a=mom.loc[dt] if br.loc[dt]>=th else -mom.loc[dt]; b=fw.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:rows.append(a[ok].corr(b[ok],method='spearman'))
 q=pd.Series(rows).dropna();print('threshold',th,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),6),'hit',round((q>0).mean(),4),'recent365',round(q.tail(365).mean(),6),'recent180',round(q.tail(180).mean(),6),'recent60',round(q.tail(60).mean(),6))
