import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in(get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except:pass
p=pd.DataFrame({s:g(s).set_index('date').close for s in U}).sort_index();r=p.pct_change(); br=(r>0).mean(1); fr=p.shift(-1)/p-1
for q in [.1,.15,.2,.25,.3]:
 for w in [2,3,5]:
  st=(br<=br.rolling(60,min_periods=30).quantile(q)).shift(1); f=(-r.rolling(w).sum()).where(st); f=f.sub(f.median(1),axis=0); a=[]; ns=[]
  for d in f.index:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
  a=np.array(a); print(q,w,len(a),round(a.mean(),5),round(a.mean()/a.std(ddof=1),5) if len(a)>1 else np.nan,round(st.sum()))
