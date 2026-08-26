import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}; cut=pd.Timestamp('2032-04-30'); H=[1,5,10,20]; R={h:[] for h in H}; cov=[]; turn=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cut]) for x in D.values()]))
for t in Ds:
 a={}; fw={h:{} for h in H}
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float); lr=np.log(c).diff()
  if len(c)<45: continue
  v=lr.tail(20).std();
  if not np.isfinite(v) or v<1e-8: continue
  # 3-session contrarian move, volatility-normalized, with gap/close returns only
  sig=-lr.tail(3).sum()/v
  a[s]=sig; rows.append({'date':t.date(),'symbol':s,'signal':sig}); fut=x.loc[x.index>t].close
  for h in H:
   if len(fut)>=h: fw[h][s]=fut.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  cov.append(len(a)/15); r=pd.Series(a).rank(pct=True); turn.append(0 if prev is None else (r-prev.reindex(r.index).fillna(.5)).abs().mean()); prev=r
  for h in H:
   co=[s for s in a if s in fw[h]]
   if len(co)>=8:
    q=spearmanr([a[s] for s in co],[fw[h][s] for s in co]).statistic
    if np.isfinite(q): R[h].append(q)
print('cutoff',cut.date(),'dates',len(Ds),'coverage',np.mean(cov),'turnover',np.mean(turn))
for h,v in R.items():
 q=pd.Series(v); print('H',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
pd.DataFrame(rows).to_csv('scripts/miner_2_20320503_short_reversal3_signal.csv',index=False)
