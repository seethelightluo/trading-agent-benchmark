import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<100:d=get_index_daily_data(s,1800)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();R=P.pct_change();rows=[];sig=[]
for t in range(80,len(P)-11):
 win=R.iloc[t-59:t+1];m=win.mean(axis=1);v={}
 for s in P:
  x=win[s];z=m;ok=x.notna()&z.notna()
  if ok.sum()<45:continue
  xx=x[ok].values;zz=z[ok].values; beta=np.cov(xx,zz,ddof=1)[0,1]/(np.var(zz,ddof=1)+1e-12)
  resid=R[s].iloc[t-2:t+1]-beta*m.iloc[-3:];iv=R[s].iloc[t-19:t+1].std()
  if len(resid.dropna())<3 or not np.isfinite(iv) or iv<=1e-8:continue
  v[s]=-resid.sum()/iv
 if len(v)>=8:sig.append(pd.Series(v,name=P.index[t]))
 for h in (1,5,10):
  q=pd.concat([pd.Series(v),R.iloc[t+1:t+h+1].sum().reindex(v)],axis=1).dropna()
  if len(q)>=8:rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 a=o[o.h==h].set_index('date').ic;print('h',h,'dates',len(a),'avgN',round(o[o.h==h].n.mean(),3),'coverage',round(o[o.h==h].n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for c in ['2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c];print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
S=pd.DataFrame(sig);S.to_csv('scripts/miner_3_20291213_residual_reversal_signal.csv',index_label='date');print('signal_rows',len(S),'assets',len(P.columns),'turnover',np.nanmean((S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)).values))
