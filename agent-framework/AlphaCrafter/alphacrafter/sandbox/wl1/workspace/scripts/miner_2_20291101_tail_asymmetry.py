import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2029-10-31'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base=Path('../persistent/stock_data')
close={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cutoff] for s in syms}; idx=sorted(set().union(*[x.index for x in close.values()])); p=pd.DataFrame({s:close[s].reindex(idx) for s in syms},index=idx); r=p.pct_change()
breadth=r.gt(0).rolling(10,min_periods=5).mean().mean(axis=1); gate=0.5+0.8*(breadth-0.5).clip(-.5,.5)
sig=pd.DataFrame(index=idx,columns=syms,dtype=float)
for s,x in close.items():
 rr=x.pct_change(); ret20=x.pct_change(20); down=rr.where(rr<0).rolling(40,min_periods=5).std(); up=rr.where(rr>0).rolling(40,min_periods=5).std(); asym=((up+1e-8)/(down+1e-8)).clip(.5,2)
 sig[s]=(ret20/(down*np.sqrt(20)+1e-8)*asym).reindex(idx)
sig=sig.mul(gate,axis=0).shift(1)
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in idx:
  ok=sig.loc[d].notna()&fwd.loc[d].notna()
  if ok.sum()>=8: vals.append(spearmanr(sig.loc[d,ok],fwd.loc[d,ok]).statistic); ns.append(ok.sum())
 z=np.asarray(vals); print('H %d dates %d IC %.6f ICIR %.6f avgN %.2f hit %.4f'%(h,len(z),np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(ns),np.mean(z>0)))
 if h==1:
  for name,mask in [('2020-25',sig.index<'2026-01-01'),('2026+',sig.index>='2026-01-01'),('2028+',sig.index>='2028-01-01'),('2029YTD',sig.index>='2029-01-01')]:
   q=[]
   for d in sig.index[mask]:
    ok=sig.loc[d].notna()&fwd.loc[d].notna()
    if ok.sum()>=8:q.append(spearmanr(sig.loc[d,ok],fwd.loc[d,ok]).statistic)
   q=np.asarray(q); print(name,'dates',len(q),'IC %.6f ICIR %.6f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1)))
ranks=sig.rank(axis=1,pct=True); print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),ranks.diff().abs().mean(axis=1).dropna().mean())); sig.index.name='date'; sig.to_csv('scripts/miner_2_20291101_tail_asymmetry_signal.csv')
