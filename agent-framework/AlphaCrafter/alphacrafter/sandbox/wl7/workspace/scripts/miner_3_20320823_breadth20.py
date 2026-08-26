import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
cut=pd.Timestamp('2032-08-22'); H=[1,5,10,20]; R={h:[] for h in H}; rows=[]; turns=[]; cov=[]; prev=None
Ds=sorted(set().union(*[set(x.index[x.index<=cut]) for x in D.values()]))
for t in Ds:
 breadth=[]
 for s,x in D.items():
  if t in x.index:
   c=x.loc[:t].close.astype(float)
   if len(c)>=90: breadth.append(c.iloc[-1]/c.iloc[-21]-1)
 gate=np.nanmedian(breadth) if breadth else np.nan; sig={}; fut={h:{} for h in H}
 for s,x in D.items():
  if t not in x.index: continue
  c=x.loc[:t].close.astype(float); lr=np.log(c/c.shift(1)).dropna()
  if len(c)<90: continue
  vol=lr.iloc[-60:].std()
  if not np.isfinite(vol) or vol<=0: continue
  r20=c.iloc[-1]/c.iloc[-21]-1
  sig[s]=(r20 if gate>0 else -r20)/(vol*np.sqrt(20)); rows.append({'date':t.date(),'symbol':s,'signal':sig[s]})
  for h in H:
   fx=x.loc[x.index>t].close
   if len(fx)>=h: fut[h][s]=fx.iloc[h-1]/c.iloc[-1]-1
 if len(sig)>=8:
  cov.append(len(sig)/15); rr=pd.Series(sig).rank(pct=True); turns.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in H:
   q=[s for s in sig if s in fut[h]]
   if len(q)>=8:
    v=spearmanr([sig[s] for s in q],[fut[h][s] for s in q]).statistic
    if np.isfinite(v): R[h].append(v)
print('candidate=breadth_conditional_20d_momentum_reversal cutoff',cut.date(),'universe',len(U))
print('coverage',np.mean(cov),'turnover',np.mean(turns),'avgN',np.mean([len([r for r in rows if r['date']==d]) for d in set(r['date'] for r in rows)]))
for h in H:
 q=pd.Series(R[h]); print('H',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'thirds',[q.iloc[i*len(q)//3:(i+1)*len(q)//3].mean() for i in range(3)])
pd.DataFrame(rows).to_csv('scripts/miner_3_20320823_breadth20_signal.csv',index=False)
