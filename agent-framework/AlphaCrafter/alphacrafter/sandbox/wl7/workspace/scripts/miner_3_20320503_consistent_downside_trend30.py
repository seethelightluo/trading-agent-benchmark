import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cutoff=pd.Timestamp('2032-04-29'); H=[1,5,10,20]; R={h:[] for h in H}; ns=[]; cov=[]; turn=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in Ds:
 a={}; F={h:{} for h in H}
 for s,x in D.items():
  if t not in x.index: continue
  c=x.loc[:t].close.astype(float); r=np.log(c).diff().dropna()
  if len(r)<65: continue
  z=r.tail(30); down=z[z<0]
  dd=np.sqrt(np.mean(down.values**2)) if len(down)>=3 else np.nan
  if not np.isfinite(dd) or dd<1e-8: continue
  # trend quality: return per downside risk, rewarded for directional consistency
  a[s]=(z.sum()/dd)*(abs(z).sum()/max(np.abs(z.sum()),1e-8))*((z>0).mean()-(z<0).mean()+1)/2
  rows.append({'date':t.date(),'symbol':s,'signal':a[s]})
  for h in H:
   f=x.loc[x.index>t].close
   if len(f)>=h:F[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  ns.append(len(a));cov.append(len(a)/15); q=pd.Series(a).rank(pct=True);turn.append(0 if prev is None else (q-prev.reindex(q.index).fillna(.5)).abs().mean());prev=q
  for h in H:
   co=[s for s in a if s in F[h]]
   if len(co)>=8:
    v=spearmanr([a[s] for s in co],[F[h][s] for s in co]).statistic
    if np.isfinite(v):R[h].append(v)
print('cutoff',cutoff.date(),'universe',len(U),'dates',len(Ds),'coverage',np.mean(cov),'turnover',np.mean(turn),'avgN',np.mean(ns))
for h,v in R.items():
 q=pd.Series(v);print('H',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
pd.DataFrame(rows).to_csv('scripts/miner_3_20320503_consistent_downside_trend30_signal.csv',index=False)
