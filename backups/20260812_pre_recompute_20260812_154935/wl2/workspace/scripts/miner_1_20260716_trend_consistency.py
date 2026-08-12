import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-07-15')
px={}
for s in U:
    f='../persistent/stock_data/'+s+'.csv'
    if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
    d=pd.read_csv(f,parse_dates=['date']).sort_values('date')
    d=d[d.date<=end][['date','close']].dropna().drop_duplicates('date')
    px[s]=d.set_index('date').close

# One interpretable candidate: 20-day trend consistency, fraction of positive daily returns.
# Higher consistency should predict stronger next returns; all information is through t.
dates=sorted(set().union(*[set(x.index) for x in px.values()]))
ics={h:[] for h in [1,5,10]}; turnovers=[]; valid_counts=[]
prev=None
for i,t in enumerate(dates):
    fac={}; fw={}
    for s,p in px.items():
        if t not in p.index: continue
        hist=p.loc[:t].tail(21)
        if len(hist)<21: continue
        r=hist.pct_change().dropna()
        fac[s]=float((r>0).mean())
        for h in [1,5,10]:
            fut=p.loc[p.index>t].head(h)
            if len(fut)>=h: fw.setdefault(h,{})[s]=float(fut.iloc[h-1]/p.loc[t]-1)
    valid_counts.append(len(fac))
    ranks=pd.Series(fac).rank(pct=True)
    if prev is not None:
        ss=set(prev)&set(fac)
        if len(ss)>=8: turnovers.append(float(np.mean([abs(ranks[s]-prev[s]) for s in ss])))
    prev=ranks
    for h in [1,5,10]:
        q=pd.Series(fw.get(h,{})); z=pd.Series(fac)
        common=z.index.intersection(q.index)
        if len(common)>=8:
            ics[h].append(float(spearmanr(z[common],q[common]).statistic))

def stat(a):
    a=np.asarray(a,float); return np.mean(a), np.mean(a)/np.std(a,ddof=1), np.mean(a>0),len(a)
print('dates',len(dates),'instruments',len(U),'mean_valid',np.mean(valid_counts),'coverage',np.mean(valid_counts)/len(U))
for h,a in ics.items(): print('horizon',h,'mean/icir/hit/n',stat(a))
print('turnover_rank_mean_abs_change',np.mean(turnovers),'turnover_obs',len(turnovers))
# Compare pooled rank values against existing factor expressions reproduced here only for correlation report.
# Use date cross-sectional demeaned series to avoid asset-level scale effects.
allv={}
for s,p in px.items():
    q=p.pct_change().rolling(20).apply(lambda x: np.mean(x>0),raw=True)
    allv[s]=q
# library correlations are calculated on overlapping date-asset ranks for known simple factors
libs={}
for name,kind in [('short_term_reversal_5d','rev5'),('peer_median_leadlag_5d','lead5'),('miner_2_risk_adjusted_momentum_20d','ram20')]:
  vals=[]
  for t in dates:
    vv={}
    for s,p in px.items():
      if t not in p.index: continue
      h=p.loc[:t]
      if kind=='rev5' and len(h)>=6: vv[s]=-float(h.iloc[-1]/h.iloc[-6]-1)
      if kind=='lead5' and len(h)>=6:
        rr={x:float(px[x].loc[:t].iloc[-1]/px[x].loc[:t].iloc[-6]-1) for x in px if t in px[x].index and len(px[x].loc[:t])>=6}
        if s in rr: vv[s]=rr[s]-np.median(list(rr.values()))
      if kind=='ram20' and len(h)>=21:
        rr=float(h.iloc[-1]/h.iloc[-21]-1); vol=float(h.pct_change().tail(20).std())
        if vol>0: vv[s]=rr/vol
    for s in set(vv)&set(allv):
      if t in allv[s].index and pd.notna(allv[s].loc[t]): vals.append((allv[s].loc[t],vv[s]))
  if len(vals)>100: libs[name]=float(spearmanr(pd.DataFrame(vals)[0],pd.DataFrame(vals)[1]).statistic)
print('library_corr',libs,'max_abs',max([abs(x) for x in libs.values()] or [0.0]))
