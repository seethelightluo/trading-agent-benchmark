import os, json
import numpy as np
import pandas as pd

CUT=pd.Timestamp('2032-04-01')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(sym):
    x=pd.read_csv('../persistent/stock_data/'+sym+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
    x=x[x.index<=CUT]
    return x[~x.index.duplicated(keep='last')].sort_index()
px=pd.concat({a:fetch(a) for a in assets},axis=1).sort_index()
ret=px.pct_change()
defensive=ret[['XAU','US10Y','CN10Y']].mean(axis=1)
# factor observable at t, forward return t+1:t+10 (no lookahead)
rows=[]; sigrows=[]
for t in px.index:
    if t not in ret.index: continue
    vals={}
    for a in assets:
        r=ret[a]
        if len(r.loc[:t])<65: continue
        rel25=(px[a].loc[t]/px[a].shift(25).loc[t]-1) - (px[['XAU','US10Y','CN10Y']].loc[t]/px[['XAU','US10Y','CN10Y']].shift(25).loc[t]-1).mean()
        rel30=(px[a].loc[t]/px[a].shift(30).loc[t]-1) - (px[['XAU','US10Y','CN10Y']].loc[t]/px[['XAU','US10Y','CN10Y']].shift(30).loc[t]-1).mean()
        vol=r.loc[:t].iloc[-61:-1].std()*np.sqrt(252)
        if pd.notna(rel25) and pd.notna(rel30) and pd.notna(vol) and vol>1e-8:
            vals[a]=0.5*rel25/vol+0.5*rel30/vol
    future=(px.shift(-10)/px-1).loc[t]
    common=[a for a in assets if a in vals and pd.notna(future.get(a))]
    if len(common)>=8:
        f=pd.Series({a:vals[a] for a in common}); y=future[common]
        ic=f.rank().corr(y.rank())
        rows.append((t,ic,len(common)))
        sigrows += [(t,a,vals[a]) for a in common]
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mean=ic.ic.mean(); sd=ic.ic.std(ddof=1); icir=mean/sd if sd else np.nan
# turnover based on rank ordering / signal changes, average rank displacement
s=pd.DataFrame(sigrows,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal').sort_index()
ranks=s.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).mean()
# periods
print(json.dumps({'dates':len(ic),'avg_instruments':ic.n.mean(),'coverage':len(s.stack())/(len(ic)*15),'ic_10d':mean,'icir_daily':icir,'hit_ratio':(ic.ic>0).mean(),'turnover':turnover,'period_start':str(ic.index.min().date()),'period_end':str(ic.index.max().date())},default=str))
for days in [60,180,365]:
 z=ic.tail(days); print('recent',days,'ic',z.ic.mean(),'icir',z.ic.mean()/z.ic.std(ddof=1),'dates',len(z))
os.makedirs('scripts',exist_ok=True)
s.to_csv('scripts/miner_2_20320401_defrel_blend2530_signal.csv')
ic.to_csv('scripts/miner_2_20320401_defrel_blend2530_ic.csv')
