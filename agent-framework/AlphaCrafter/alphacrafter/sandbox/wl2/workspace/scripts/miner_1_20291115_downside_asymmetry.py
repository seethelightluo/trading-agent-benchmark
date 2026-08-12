import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<100: d=get_index_daily_data(s,1800)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
rows=[]; sig=[]
# Downside-asymmetry reversal: recent return is rewarded when downside volatility is
# contained, while assets with severe negative-tail participation are penalized.
for t in range(65,len(P)-11):
 v={}
 for s in P:
  x=R[s].iloc[t-59:t+1].dropna()
  r20=R[s].iloc[t-19:t+1].dropna()
  if len(x)>=45 and len(r20)>=15:
   down=x[x<0]
   if len(down)>=5:
    dv=np.sqrt(np.mean(down.values**2)); rv=x.std(ddof=1)
    # medium horizon recovery, normalized by downside risk and tempered by total risk
    v[s]=np.tanh(r20.sum()/0.12)/(dv*np.sqrt(252)+1e-6) * (0.5+0.5*np.tanh(rv/0.04))
 for h in (1,5,10):
  q=pd.concat([pd.Series(v),R.iloc[t+1:t+h+1].sum().reindex(v)],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(pd.Series(v,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 a=o[o.h==h].set_index('date').ic
 print('h',h,'dates',len(a),'avgN',o[o.h==h].n.mean(),'coverage',o[o.h==h].n.mean()/len(U),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for c in ['2027-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_1_20291115_downside_asymmetry_signal.csv',index_label='date')
a=o[o.h==1].set_index('date').ic
print('turnover',np.nanmean((S.diff().abs().sum(1)/(S.abs().sum(1)+1e-9)).values))
print('maxabs_library_correlation_unavailable',np.nan)
