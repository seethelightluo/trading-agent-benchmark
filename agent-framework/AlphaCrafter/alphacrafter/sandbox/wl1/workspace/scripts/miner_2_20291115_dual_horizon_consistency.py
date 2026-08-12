import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2029-11-14'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base=Path('../persistent/stock_data')
close={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cutoff] for s in syms}; idx=sorted(set().union(*[x.index for x in close.values()])); p=pd.DataFrame({s:close[s].reindex(idx) for s in syms},index=idx); r=p.pct_change()
# Dual horizon trend consistency: medium trend, long trend, and path quality; all information lagged.
sig=pd.DataFrame(index=idx,columns=syms,dtype=float)
for s in syms:
 x=close[s]; rr=x.pct_change(); ret20=x.pct_change(20); ret60=x.pct_change(60); vol=rr.rolling(20,min_periods=10).std(); dd=x/x.rolling(60,min_periods=30).max()-1
 path=(ret20.abs()/(rr.abs().rolling(20,min_periods=10).sum()+1e-8)).clip(0,2)
 # reward aligned positive trends, suppress drawdown and unstable paths
 consistency=(np.sign(ret20)*np.sign(ret60)).replace(0,1)
 sig[s]=((0.6*ret20/(vol*np.sqrt(20)+1e-8)+0.4*ret60/(rr.rolling(60,min_periods=30).std()*np.sqrt(60)+1e-8))* (0.7+0.3*path) * consistency * (1+dd.clip(-.5,0))).reindex(idx)
sig=sig.shift(1)
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in idx:
  ok=sig.loc[d].notna()&fwd.loc[d].notna()
  if ok.sum()>=8: vals.append(spearmanr(sig.loc[d,ok],fwd.loc[d,ok]).statistic); ns.append(ok.sum())
 z=np.asarray(vals); print('H',h,'dates',len(z),'IC %.6f ICIR %.6f avgN %.2f hit %.4f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(ns),np.mean(z>0)))
 if h==1:
  for name,mask in [('2020-25',sig.index<'2026-01-01'),('2026+',sig.index>='2026-01-01'),('2028+',sig.index>='2028-01-01'),('2029YTD',sig.index>='2029-01-01')]:
   q=[]
   for d in sig.index[mask]:
    ok=sig.loc[d].notna()&fwd.loc[d].notna()
    if ok.sum()>=8:q.append(spearmanr(sig.loc[d,ok],fwd.loc[d,ok]).statistic)
   q=np.asarray(q); print(name,'dates',len(q),'IC %.6f ICIR %.6f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1)))
ranks=sig.rank(axis=1,pct=True); print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),ranks.diff().abs().mean(axis=1).dropna().mean())); sig.index.name='date'; sig.to_csv('scripts/miner_2_20291115_dual_horizon_consistency_signal.csv')
