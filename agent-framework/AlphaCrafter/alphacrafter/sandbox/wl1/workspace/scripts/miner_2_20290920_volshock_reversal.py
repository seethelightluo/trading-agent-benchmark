import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2029-09-19')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); close={}
for s in syms:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 close[s]=d[d.index<=cutoff]
idx=sorted(set().union(*[x.index for x in close.values()]))
p=pd.DataFrame({s:close[s].reindex(idx) for s in syms},index=idx)
# Volatility-shock reversal: recent weakness is favored only when short-term vol has not exploded;
# cross-sectional ranks make the signal robust across heterogeneous asset scales.
sig=pd.DataFrame(index=idx,columns=syms,dtype=float)
for s in syms:
 x=close[s]; r=x.pct_change()
 ret5=x.pct_change(5); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
 shock=(v20/(v60+1e-8)).clip(0.25,4)
 # contrarian short-term reversal, penalized by volatility shock and absolute risk
 raw=(-ret5/(v20+1e-8))/(1+0.8*(shock-1).clip(lower=0))
 sig[s]=raw.reindex(idx)
# lag to ensure no lookahead
sig=sig.shift(1)
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   vals.append(spearmanr(sig.loc[dt,ok],fwd.loc[dt,ok]).statistic); ns.append(ok.sum())
 vals=np.asarray(vals); ic=np.nanmean(vals); ir=ic/np.nanstd(vals,ddof=1)*np.sqrt(252)
 print(f'H {h} dates {len(vals)} IC {ic:.6f} ICIR {ir:.6f} avgN {np.mean(ns):.2f} hit {np.mean(vals>0):.4f}')
 if h==10:
  for name,mask in [('2020-25',sig.index<'2026-01-01'),('2026+',sig.index>='2026-01-01'),('2028+',sig.index>='2028-01-01'),('2029YTD',sig.index>='2029-01-01')]:
   x=[]
   for dt in sig.index[mask]:
    ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
    if ok.sum()>=8:x.append(spearmanr(sig.loc[dt,ok],fwd.loc[dt,ok]).statistic)
   x=np.asarray(x); print(name,'dates',len(x),'IC %.6f ICIR %.6f'%(np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1)*np.sqrt(252)))
ranks=sig.rank(axis=1,pct=True)
print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),ranks.diff().abs().mean(axis=1).dropna().mean()))
sig.index.name='date';sig.to_csv('scripts/miner_2_20290920_volshock_reversal_signal.csv')
