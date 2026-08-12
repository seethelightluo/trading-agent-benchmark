import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None and len(d): px[s]=d.set_index('date')
P=pd.DataFrame({s:d.close.astype(float) for s,d in px.items()}).sort_index(); R=P.pct_change()
rows=[]; sig=[]
# Volatility-breadth quality: low idiosyncratic recent volatility is preferred,
# but only when its medium-horizon return is not deeply negative. This interpretable
# combination aims to avoid distressed/high-volatility assets without using future data.
for t in range(65,len(P)-11):
 f={}
 for s in P:
  r=R[s].iloc[:t+1].dropna()
  if len(r)<60: continue
  v20=r.iloc[-20:].std(); v60=r.iloc[-60:].std()
  mom=r.iloc[-20:].sum()
  if not np.isfinite(v20) or not np.isfinite(v60) or v20<=1e-8: continue
  # low volatility, with a modest positive-return quality tilt; cross-sectional rank handles scales
  f[s]=-(0.65*np.log(v20)+0.35*np.log(v60)) + 0.20*(mom/(v60+1e-9))
 f=pd.Series(f).dropna()
 if len(f)<8: continue
 f=f.rank(pct=True)-0.5; sig.append(f.rename(P.index[t]))
 for h in (1,5,10):
  fw=R.iloc[t+1:t+1+h].sum().reindex(f.index)
  q=pd.concat([f,fw],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 z=o[o.h==h]; a=z.set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for c in ['2025-01-01','2028-01-01','2029-01-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_1_20300321_volatility_quality_signal.csv',index_label='date')
print('signal_rows',len(S),'turnover',np.nanmean((S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)).values),'instruments',len(U),'available',len(px))
