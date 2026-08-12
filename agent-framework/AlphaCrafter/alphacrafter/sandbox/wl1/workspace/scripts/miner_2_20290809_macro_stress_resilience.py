import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

cutoff=pd.Timestamp('2029-08-08')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data')
prices={}
for s in syms:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 prices[s]=d[d.index<=cutoff]
p=pd.DataFrame(prices).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(p.index).ffill()
# Macro-conditioned downside-resilient relative momentum. VIX stress is lagged and cross-sectionally
# changes the reward for positive momentum vs weak assets; all inputs end-of-day and signal is lagged.
r20=p.pct_change(20); r1=p.pct_change(); down=r1.clip(upper=0).rolling(20,min_periods=10).std()*np.sqrt(252)
# VIX stress based only on trailing 60d percentile, capped; no lookahead
stress=(vix-vix.rolling(60).median())/(vix.rolling(60).std()+1e-9)
stress=stress.clip(-1,2).fillna(0)
rel=r20.sub(r20.median(axis=1),axis=0)
base_sig=rel/(down+0.02)
# stress rewards positive relative momentum and penalizes negative relative momentum asymmetrically
sig=base_sig*(1+0.35*stress.values[:,None]*np.tanh(rel*5))
sig=sig.shift(1)
rows=[]
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1
 ics=[]; turns=[]; counts=[]
 for dt in sig.index:
  a=sig.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   ics.append(spearmanr(a[ok],b[ok]).statistic); counts.append(ok.sum())
 # rank turnover at successive observations
 ranks=sig.rank(axis=1,pct=True); turns=ranks.diff().abs().mean(axis=1).dropna()
 ic=np.nanmean(ics); sd=np.nanstd(ics,ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
 rows.append((h,len(ics),ic,icir,np.mean(counts),turns.mean()))
print('cutoff',cutoff.date(),'dates',len(p),'avg instruments',p.notna().sum(axis=1).mean())
for x in rows: print('H',x[0],'dates',x[1],'IC %.6f ICIR %.6f avgN %.2f turnover %.6f'%x[2:])
# regime 10d
fwd=p.shift(-10)/p-1
for name,mask in [('2020-25',sig.index<'2026-01-01'),('2026+',sig.index>='2026-01-01'),('2028+',sig.index>='2028-01-01'),('2029YTD',sig.index>='2029-01-01')]:
 xs=[]
 for dt in sig.index[mask]:
  a=sig.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: xs.append(spearmanr(a[ok],b[ok]).statistic)
 print(name,'dates',len(xs),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(xs),np.mean(xs)/np.std(xs,ddof=1)*np.sqrt(252),np.mean(np.array(xs)>0)))
# artifact
out=pd.DataFrame(sig); out.index.name='date'; out.to_csv('scripts/miner_2_20290809_macro_stress_resilience_signal.csv')
print('coverage',sig.notna().mean().mean())
