import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: recovery-adjusted medium momentum. Reward 60d trend, penalize unresolved drawdown,
# with volatility normalization; all inputs lagged at date t and forward return starts t+1.
px={}
for s in U:
    d=get_stock_daily_data(s, days=6000)
    if d is None or len(d)<300: continue
    x=d[['date','close']].copy().dropna().drop_duplicates('date').set_index('date')['close']
    px[s]=x
P=pd.DataFrame(px).sort_index()
# broad date range and aligned returns
r=P.pct_change()
rows=[]
for t in P.index[120:-12]:
    vals={}; fwd={}
    for s in U:
        if s not in P: continue
        z=P[s].loc[:t].tail(121)
        if len(z)<121: continue
        rr=r[s].loc[:t].tail(60)
        if rr.isna().any(): continue
        # recovery-adjusted trend: 60d return + recovery from 120d trough, scaled by 20d risk
        mom=z.iloc[-1]/z.iloc[-61]-1
        dd=z.iloc[-1]/z.max()-1
        rec=z.iloc[-1]/z.tail(120).min()-1
        vol=r[s].loc[:t].tail(20).std()
        if not np.isfinite(vol) or vol<=0: continue
        vals[s]=(0.65*mom+0.35*rec-0.25*abs(dd))/vol
        future=P[s].loc[t:].iloc[1:11]
        if len(future)>=10 and future.notna().all(): fwd[s]=future.iloc[-1]/P[s].loc[t]-1
    common=list(set(vals)&set(fwd))
    if len(common)>=8:
        a=pd.Series({s:vals[s] for s in common}); b=pd.Series({s:fwd[s] for s in common})
        ic=a.corr(b,method='spearman')
        rows.append((t,ic,len(common)))
out=pd.DataFrame(rows,columns=['date','ic','n']).dropna()
print('dates',len(out),'avg_n',out.n.mean(),'coverage',out.n.sum()/(len(out)*len(U)))
print('IC',out.ic.mean(),'ICIR',out.ic.mean()/out.ic.std(ddof=1),'hit',(out.ic>0).mean())
for start in ['2020-01-01','2025-01-01','2030-01-01','2034-01-01']:
 q=out[out.date>=pd.Timestamp(start)]
 print(start,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
print('turnover proxy: rank signal changes not calculated; decay horizons omitted in candidate screen')
# artifact with reproducible values for latest dates
sig=[]
for t in P.index[-80:-12]:
 for s in U:
  if s not in P: continue
  z=P[s].loc[:t].tail(121)
  if len(z)<121: continue
  vol=r[s].loc[:t].tail(20).std()
  if vol>0:
   mom=z.iloc[-1]/z.iloc[-61]-1; dd=z.iloc[-1]/z.max()-1; rec=z.iloc[-1]/z.tail(120).min()-1
   sig.append((t,s,(.65*mom+.35*rec-.25*abs(dd))/vol))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('../persistent/miner_1_20350706_recovery_adjusted_momentum_signal.csv',index=False)
