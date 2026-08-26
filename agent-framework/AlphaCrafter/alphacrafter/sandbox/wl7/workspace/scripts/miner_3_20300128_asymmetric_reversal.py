import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
TODAY='2030-01-28'; Hs=[1,5,10,20]
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 try:d=get_index_daily_data(s,days=4000)
 except: d=None
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,days=4000)
  except:d=None
 if d is not None and len(d):px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]; rows={h:[] for h in Hs}
for i in range(80,len(P)-max(Hs)):
 vals=[]
 for s in u:
  if s not in P:continue
  x=P[s].iloc[:i+1].dropna()
  if len(x)<40:continue
  rr=x.pct_change().iloc[-21:-1]; down=rr[rr<0].std(); allv=rr.std()
  if not np.isfinite(down) or down<=0:continue
  # asymmetric short reversal: recent loss is more predictive when downside risk is high
  f=-(x.iloc[-1]/x.iloc[-4]-1)/down
  for h in Hs:
   if i+h<len(P) and pd.notna(P[s].iloc[i+h]):rows[h].append((P.index[i],s,f,P[s].iloc[i+h]/x.iloc[-1]-1))
for h in Hs:
 df=pd.DataFrame(rows[h],columns=['date','symbol','factor','fwd']).dropna(); ic=df.groupby('date').apply(lambda z:z.factor.corr(z.fwd),include_groups=False).dropna()
 print('horizon',h,'dates',len(ic),'avg_names',round(df.groupby('date').size().mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4),'coverage',round(df.symbol.nunique()/len(u),4))
 if h==10:
  r=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',r.diff().abs().mean(axis=1).dropna().mean());df.to_csv('scripts/miner_3_20300128_asymmetric_reversal_signal.csv',index=False)
