import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<100: d=get_index_daily_data(s,2600)
 if d is not None: close[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(close).sort_index(); R=P.pct_change()
# Defensive-confirmed residual reversal: short-term reversal is strongest when
# the lagged cross-asset defensive basket (XAU, US10Y, CN10Y) is weak/strong;
# use only information through t and scale by cross-sectional stress dispersion.
defs=[x for x in ['XAU','US10Y','CN10Y'] if x in P]
rows=[]; sig=[]
for t in range(80,len(P)-11):
 rr=R.iloc[t-2:t+1].sum(); med=R.iloc[t-2:t+1].median(axis=1).sum()
 vol=R.iloc[t-19:t+1].std()
 # stress = recent defensive basket return minus broad median return, trailing only
 db=R.iloc[t-4:t+1][defs].mean(axis=1).sum() if defs else 0
 broad=R.iloc[t-4:t+1].median(axis=1).sum()
 stress=np.clip(1.0 + 2.0*(db-broad),0.5,1.5)
 f=(-(rr-med)/vol.replace(0,np.nan))*stress
 f=f.replace([np.inf,-np.inf],np.nan).dropna(); sig.append(f.rename(P.index[t]))
 for h in (1,3,5,10):
  fw=R.iloc[t+1:t+h+1].sum().reindex(f.index); q=pd.concat([f,fw],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,3,5,10):
 z=o[o.h==h]; a=z.set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2030')]:
  b=a[(a.index.astype(str)>=lo+'-01-01')&(a.index.astype(str)<=hi+'-12-31')]
  print(lo+'-'+hi,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6))
S=pd.DataFrame(sig); S.to_csv('scripts/miner_2_20301017_defensive_confirmed_reversal_signal.csv',index_label='date')
print('signal_rows',len(S),'instruments',len(U),'available',len(close),'defensive',defs)
# turnover proxy: mean rank changes across adjacent signal dates
if len(S)>1:
 print('turnover_proxy',round(S.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
print('max_abs_library_correlation',None)
