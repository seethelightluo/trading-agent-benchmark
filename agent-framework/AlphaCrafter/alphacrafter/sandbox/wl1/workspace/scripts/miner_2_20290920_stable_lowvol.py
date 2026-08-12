import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2029-09-19'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base=Path('../persistent/stock_data')
close={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cutoff] for s in syms}; idx=sorted(set().union(*[x.index for x in close.values()])); p=pd.DataFrame({s:close[s].reindex(idx) for s in syms},index=idx)
# Stable low-volatility quality: inverse medium volatility, rewarded when short vol is not accelerating.
sig=pd.DataFrame(index=idx,columns=syms,dtype=float)
for s,x in close.items():
 r=x.pct_change(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std(); accel=(v20/(v60+1e-8)).clip(.25,4)
 sig[s]=(1/(v60+1e-8)/(1+0.7*(accel-1).clip(lower=0))).reindex(idx)
sig=sig.shift(1)
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ns=[]
 for d in idx:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.array(z); print('H',h,'dates',len(z),'IC %.6f ICIR %.6f avgN %.2f hit %.4f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1)*np.sqrt(252),np.mean(ns),np.mean(z>0)))
 if h==10:
  for n,m in [('2020-25',sig.index<'2026-01-01'),('2026+',sig.index>='2026-01-01'),('2028+',sig.index>='2028-01-01'),('2029YTD',sig.index>='2029-01-01')]:
   q=[]
   for d in sig.index[m]:
    ok=sig.loc[d].notna()&f.loc[d].notna()
    if ok.sum()>=8:q.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic)
   q=np.array(q);print(n,'dates',len(q),'IC %.6f ICIR %.6f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(252)))
print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()));sig.index.name='date';sig.to_csv('scripts/miner_2_20290920_stable_lowvol_signal.csv')
