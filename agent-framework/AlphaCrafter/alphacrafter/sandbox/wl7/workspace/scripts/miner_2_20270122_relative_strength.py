import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-01-22')
frames={}
for s in UNIV:
    p=os.path.join('..','persistent','stock_data',s+'.csv')
    d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
    d=d.loc[d.index<=END]
    frames[s]=d['close'].astype(float)
px=pd.concat(frames,axis=1).sort_index()
ret=px.pct_change(20)
# signal at t uses close through t-1: lagged 20d relative strength vs cross-sectional median
sig=ret.shift(1).sub(ret.shift(1).median(axis=1),axis=0)
fwd=px.pct_change(1).shift(-1)
rows=[]; byh={1:[],5:[],10:[],20:[]}
for dt in sig.index:
    for h in byh:
        future=px.pct_change(h).shift(-h).loc[dt]
        z=pd.concat([sig.loc[dt],future],axis=1).dropna()
        if len(z)>=8: byh[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
    if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
# rank turnover on consecutive valid dates
r=sig.rank(axis=1,pct=True); common=r.index.intersection(a.index)
to=r.loc[common].diff().abs().mean(axis=1).dropna().mean()
print('dates',len(a),'avg_n',a.n.mean(),'coverage',sig.notna().sum(axis=1).reindex(a.index).mean()/15)
print('ic',a.ic.mean(),'icir',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean(),'turnover',to)
print('decay', {h:(np.mean(v),np.mean(v)/np.std(v,ddof=1),len(v)) for h,v in byh.items()})
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-01-22')]:
 q=a.loc[lo:hi].ic
 print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=pd.DataFrame({'date':sig.index})
for s in UNIV: out[s]=sig[s].values
out=out[out.date.isin(a.index)]
out.to_csv('scripts/miner_2_20270122_relative_strength_signal.csv',index=False)
metrics={'daily_paper_ic':float(a.ic.mean()),'daily_paper_icir':float(a.ic.mean()/a.ic.std(ddof=1)),'ic_hit_ratio':float((a.ic>0).mean()),'valid_dates':int(len(a)),'average_instruments_per_date':float(a.n.mean()),'universe_instruments':15,'factor_coverage':float(sig.notna().sum(axis=1).reindex(a.index).mean()/15),'rank_turnover':float(to),'decay':{f'{h}d_ic':float(np.mean(v)) for h,v in byh.items()},'max_abs_library_correlation':None}
json.dump(metrics,open('scripts/miner_2_20270122_relative_strength_metrics.json','w'),indent=2)
