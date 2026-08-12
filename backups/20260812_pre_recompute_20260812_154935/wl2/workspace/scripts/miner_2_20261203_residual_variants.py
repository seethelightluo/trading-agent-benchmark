import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-03')
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for a in A}).sort_index(); r=p.pct_change(); m=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close.reindex(r.index).ffill().pct_change()
for ml,bl in [(30,60),(40,60),(30,40)]:
 out={h:[] for h in [1,5,10]}; sig=[]
 for i in range(bl+2,len(r)-10):
  rr=r.iloc[i-bl:i]; mm=m.iloc[i-bl:i]
  if mm.notna().sum()<bl*.7: continue
  v=mm.var(); b=rr.apply(lambda x:x.cov(mm)/v if v>1e-12 else np.nan); f=r.iloc[i-ml:i].sum()-b*m.iloc[i-ml:i].sum(); sig.append((r.index[i],f))
  for h in out:
   z=pd.concat([f,r.iloc[i+1:i+1+h].sum()],axis=1).dropna()
   if len(z)>=8: out[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('variant',ml,bl)
 for h,x in out.items():
  x=np.array(x); print(h,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 q=pd.DataFrame({d:f for d,f in sig}).T.rank(axis=1,pct=True); print('turn',round(q.diff().abs().mean().mean(),5),'cov',round(r.loc[q.index].notna().mean().mean(),4))
