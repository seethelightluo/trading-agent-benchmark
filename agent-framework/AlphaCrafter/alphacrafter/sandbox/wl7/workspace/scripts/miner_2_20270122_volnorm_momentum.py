import os,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-01-22')
px=pd.concat({s:pd.read_csv(os.path.join('..','persistent','stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:END]
r=px.pct_change(); vol=r.rolling(20).std(); sig=(px.pct_change(60)/vol).shift(1); fwd=px.pct_change().shift(-1)
ics=[]; hs={1:[],5:[],10:[],20:[]}
for d in sig.index:
 z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append((d,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 for h in hs:
  z=pd.concat([sig.loc[d],px.pct_change(h).shift(-h).loc[d]],axis=1).dropna()
  if len(z)>=8: hs[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
a=pd.DataFrame(ics,columns=['date','n','ic']).set_index('date'); mean=a.ic.mean(); sd=a.ic.std(ddof=1)
t=sig.rank(pct=True).loc[a.index].diff().abs().mean(axis=1).dropna().mean()
print('dates',len(a),'avg_n',a.n.mean(),'coverage',sig.notna().sum(axis=1).reindex(a.index).mean()/15)
print('ic',mean,'icir',mean/sd,'hit',(a.ic>0).mean(),'turnover',t)
print('decay',{h:(float(np.mean(v)),float(np.mean(v)/np.std(v,ddof=1)),len(v)) for h,v in hs.items()})
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-01-22')]:
 q=a.loc[lo:hi].ic; print('regime',lo,len(q),float(q.mean()),float(q.mean()/q.std(ddof=1)))
out=pd.DataFrame({'date':sig.index});
for s in U: out[s]=sig[s].values
out[out.date.isin(a.index)].to_csv('scripts/miner_2_20270122_volnorm_momentum_signal.csv',index=False)
json.dump({'daily_paper_ic':float(mean),'daily_paper_icir':float(mean/sd),'ic_hit_ratio':float((a.ic>0).mean()),'valid_dates':len(a),'average_instruments_per_date':float(a.n.mean()),'universe_instruments':15,'factor_coverage':float(sig.notna().sum(axis=1).reindex(a.index).mean()/15),'rank_turnover':float(t),'decay':{f'{h}d_ic':float(np.mean(v)) for h,v in hs.items()},'max_abs_library_correlation':None},open('scripts/miner_2_20270122_volnorm_momentum_metrics.json','w'),indent=2)
