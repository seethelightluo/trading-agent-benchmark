import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv'); d.date=pd.to_datetime(d.date); x=d.set_index('date').close.astype(float); x.name=s; px[s]=x
p=pd.concat(px.values(),axis=1).sort_index(); r20=p.pct_change(20); r60=p.pct_change(60)
sig=(r20-r20.median(axis=1)) - 0.35*(r60-r60.median(axis=1))/3
cut=sig.index<=pd.Timestamp('2032-09-15'); fwd={h:p.shift(-h).div(p)-1 for h in [5,10,20,40]}
def calc(x):
 vals=[]; ns=[]
 for dt in sig.index[cut]:
  z=pd.concat([sig.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals); return a,len(ns),np.mean(ns)
print('candidate=residual relative momentum 20d with slow-trend acceleration; universe=15 cutoff=2032-09-15')
for h,x in fwd.items():
 a,n,avg=calc(x); print('horizon',h,'dates',n,'avg_n',round(avg,3),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
print('coverage',sig.loc[cut].notna().sum().sum()/sig.loc[cut].size,'rank_turnover_proxy',sig.loc[cut].rank(axis=1,pct=True).diff().abs().mean().mean())
a,n,avg=calc(fwd[10]); rr=pd.Series(a); print('halves_IC',rr.iloc[:len(rr)//2].mean(),rr.iloc[len(rr)//2:].mean())
