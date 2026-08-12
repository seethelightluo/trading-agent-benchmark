import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(axis=1); disp=R.std(axis=1); gate=disp.rolling(20).mean()
rows=[]; signals=[]; future={1:[],5:[],10:[]}
for t in range(85,len(P)-10):
 g=disp.iloc[t]/gate.iloc[t]
 if not np.isfinite(g): continue
 v={}
 for s in P:
  z=pd.concat([R[s].iloc[t-59:t+1],m.iloc[t-59:t+1]],axis=1).dropna()
  if len(z)<20 or z.iloc[:,1].var()<=1e-12: continue
  beta=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var(); vol=z.iloc[:,0].std()
  resid=(R[s].iloc[t-4:t+1]-beta*m.iloc[t-4:t+1]).sum()
  if vol>1e-8: v[s]=-resid/vol*min(2,max(0,float(g)))
 q=pd.DataFrame({'f':pd.Series(v),'r1':R.iloc[t+1].reindex(v),'r5':P.pct_change(5).iloc[t+5].reindex(v),'r10':P.pct_change(10).iloc[t+10].reindex(v)}).dropna()
 if len(q)>=8:
  rows.append((P.index[t],len(q),q.f.corr(q.r1),q.f.corr(q.r5),q.f.corr(q.r10))); signals.append(pd.Series(v,name=P.index[t]))
out=pd.DataFrame(rows,columns=['date','n','ic1','ic5','ic10']).set_index('date')
print('assets',len(P.columns),'dates',len(out),'avgN',out.n.mean(),'coverage',out.n.mean()/len(U))
for h in ['ic1','ic5','ic10']:
 a=out[h].dropna(); print(h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for cut in ['2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=cut]; print(cut,len(b),b.mean(),b.mean()/b.std(ddof=1))
S=pd.DataFrame(signals); S.to_csv('scripts/miner_3_20290906_dispersion_residual5_signal.csv',index_label='date')
print('turnover',S.diff().abs().mean().mean())
