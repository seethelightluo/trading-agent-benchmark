import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
cut=pd.Timestamp('2032-08-08'); H=[1,5,10,20]; R={h:[] for h in H}; rows=[]; turns=[]; prev=None; cov=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cut]) for x in D.values()]))
for t in Ds:
 sig={}; fut={h:{} for h in H}; breadth=[]
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float); lr=np.log(c/c.shift(1)).dropna()
  if len(c)<65: continue
  r20=c.iloc[-1]/c.iloc[-21]-1; breadth.append(r20)
 # conditional: broad positive trend -> short-term momentum; broad negative/flat -> reversal
 gate=np.nanmedian(breadth) if breadth else np.nan
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float); lr=np.log(c/c.shift(1)).dropna()
  if len(c)<65: continue
  vol=lr.iloc[-40:].std()
  if not np.isfinite(vol) or vol<=0: continue
  r5=c.iloc[-1]/c.iloc[-6]-1
  sig[s]=(r5 if gate>0 else -r5)/(vol*np.sqrt(5))
  rows.append({'date':t.date(),'symbol':s,'signal':sig[s]})
  for h in H:
   fx=x.loc[x.index>t].close
   if len(fx)>=h: fut[h][s]=fx.iloc[h-1]/c.iloc[-1]-1
 if len(sig)>=8:
  cov.append(len(sig)/15); rr=pd.Series(sig).rank(pct=True)
  turns.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in H:
   q=[s for s in sig if s in fut[h]]
   if len(q)>=8:
    v=spearmanr([sig[s] for s in q],[fut[h][s] for s in q]).statistic
    if np.isfinite(v): R[h].append(v)
print('candidate=breadth_conditional_5d_momentum_reversal cutoff',cut.date(),'universe',len(U))
print('dates',len(R[10]),'coverage',np.mean(cov),'turnover',np.mean(turns),'avgN',len(U))
for h in H:
 q=pd.Series(R[h]); n=len(q); print('H',h,'n',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'thirds',[q.iloc[i*n//3:(i+1)*n//3].mean() for i in range(3)])
pd.DataFrame(rows).to_csv('scripts/miner_3_20320809_breadth_conditional_signal.csv',index=False)
