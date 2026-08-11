import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-11-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()]))
px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill()
r=px.pct_change()
# Defensive leadership: asset's return relative to defensive basket, gated by broad risk-off breadth.
defens=px[['XAU','US10Y','CN10Y']].pct_change(20).mean(axis=1)
market=r.mean(axis=1)
breadth=(r.rolling(20).mean()>0).mean(axis=1)
riskoff=(breadth<0.40).astype(float)
# reward assets outperforming defensive basket in risk-off, while retain a small unconditional component
raw=px.pct_change(20).sub(defens,axis=0)
f=raw.mul((0.25+0.75*riskoff),axis=0).shift(1)
print('factor defensive_relative_lead_20d universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(px.index[i])
 a=np.asarray(I); ds=pd.DatetimeIndex(ds)
 print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,start in [('2026+', '2026-01-01'),('2027+','2027-01-01'),('2028+','2028-01-01')]:
  mk=ds>=pd.Timestamp(start); q=a[mk]; print(lab,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'dates',int(mk.sum()))
rank=f.rank(axis=1,pct=True)
print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()),'riskoff_share',round(riskoff.mean(),4))
f.to_csv('scripts/miner_1_20281116_defensive_relative_lead_signal.csv',index_label='date')
