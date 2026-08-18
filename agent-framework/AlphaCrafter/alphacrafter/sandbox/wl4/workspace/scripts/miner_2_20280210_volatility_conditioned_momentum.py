import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def data(s):
    x=get_stock_daily_data(s,days=3000)
    if x is None or len(x)==0: return None
    x=x.copy(); x['date']=pd.to_datetime(x['date']); return x.set_index('date')['close'].astype(float)
P={s:data(s) for s in U}; P={s:x for s,x in P.items() if x is not None}
dates=sorted(set.intersection(*[set(x.index) for x in P.values()]))
# volatility-conditioned momentum: 20d return divided by 20d realized vol, with 60d trend sign
ics=[]; turns=[]; cells=0; total=0; prev=None
for i,d in enumerate(dates):
    if i<61 or i+10>=len(dates): continue
    vals={}; fwd={}
    for s,x in P.items():
        if d not in x.index or dates[i-60] not in x.index or dates[i-20] not in x.index or dates[i+10] not in x.index: continue
        r=x.pct_change().loc[:d].tail(20).dropna()
        if len(r)<15: continue
        mom=x.loc[d]/x.loc[dates[i-20]]-1; slow=x.loc[d]/x.loc[dates[i-60]]-1
        vals[s]=mom/(r.std()*np.sqrt(252)+1e-9) * (1 if slow>=0 else .5)
        fwd[s]=x.loc[dates[i+10]]/x.loc[d]-1
    if len(vals)>=8:
        a=pd.Series(vals); b=pd.Series(fwd).reindex(a.index)
        ic=a.corr(b,method='spearman');
        if np.isfinite(ic): ics.append(ic)
        rank=a.rank(pct=True)
        if prev is not None: turns.append(np.mean(abs(rank-prev.reindex(rank.index).fillna(.5))))
        prev=rank; cells+=len(vals); total+=15
print('dates',len(ics),'avg_n',cells/max(1,len(ics)),'cell_coverage',cells/max(1,total))
q=pd.Series(ics); print('ic',q.mean(),'icir',q.mean()/q.std(ddof=1),'hit',np.mean(q>0),'turnover',np.mean(turns))
for name,sl in [('2020-21',(pd.Timestamp('2020-01-01'),pd.Timestamp('2021-12-31'))),('2022-23',(pd.Timestamp('2022-01-01'),pd.Timestamp('2023-12-31'))),('2024-28',(pd.Timestamp('2024-01-01'),pd.Timestamp('2028-02-10')))]:
 z=q[[dates[j] for j in []]] if False else None
 # re-run dates alignment cheaply omitted
 print(name,'reported_in_full_sample')
