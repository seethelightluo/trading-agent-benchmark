# miner_3_20320722_market_downside_residual_reversal_5_20d.py
"""One candidate: beta-residual 5d reversal, active only after a broad 5d market decline."""
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CUT=pd.Timestamp('2032-07-21'); ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym):
    x=pd.read_csv(f'../persistent/stock_data/{sym}.csv',parse_dates=['date'])[['date','close']]
    return x.set_index('date').close.rename(sym)
P=pd.concat([load(s) for s in ASSETS],axis=1).sort_index().loc[:CUT]
# preserve actual shared observations; returns are only based on completed prices
R=P.pct_change(); M=R.mean(axis=1); beta=R.rolling(60).cov(M).unstack().reindex(columns=ASSETS).div(M.rolling(60).var(),axis=0)
E=R-beta.mul(M,axis=0); score=-E.rolling(5).sum().div(E.rolling(20).std())
# broad downside condition is known at signal time and deliberately does not select individual names
active=M.rolling(5).sum()<0
F=score.where(active, np.nan)

def report(h):
    fw=P.shift(-h).div(P)-1; vals=[]; ns=[]
    for d in F.index:
        a=F.loc[d]; b=fw.loc[d]; z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
    v=np.array(vals); ic=float(np.nanmean(v)); sd=float(np.nanstd(v,ddof=1)); ir=ic/sd if sd else np.nan
    print(f'H{h}: IC={ic:.6f} ICIR={ir:.6f} dates={len(v)} hit={(v>0).mean():.4%} meanN={np.mean(ns):.2f}')
    # chronological market-regime partitions
    for name,a,b in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01','2032-07-21')]:
        vv=[]
        for d in F.loc[a:b].index:
            z=pd.concat([F.loc[d],fw.loc[d]],axis=1).dropna()
            if len(z)>=8: vv.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
        vv=np.array(vv); ii=vv.mean() if len(vv) else np.nan; rr=ii/vv.std(ddof=1) if len(vv)>1 else np.nan
        print(f'  {name}: dates={len(vv)} IC={ii:.6f} ICIR={rr:.6f} hit={(vv>0).mean():.4%}' if len(vv) else f'  {name}: no dates')
    return ic,ir,len(v)
print('candidate=market-downside conditional beta-residual reversal; cutoff',CUT.date())
print('calendar dates',len(P),'assets',len(ASSETS),'active dates',int(active.sum()),'coverage',F.notna().mean().mean())
print('valid cells',int(F.notna().sum().sum()))
for h in [1,5,10,20]: report(h)
# turnover among adjacent active dates (rank correlation transformed to 1-rho)
turn=[]; prev=None
for _,x in F.dropna(how='all').iterrows():
 if prev is not None:
  z=pd.concat([prev,x],axis=1).dropna()
  if len(z)>=8: turn.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 prev=x
print('rank_turnover',np.mean(turn),'pairs',len(turn))
