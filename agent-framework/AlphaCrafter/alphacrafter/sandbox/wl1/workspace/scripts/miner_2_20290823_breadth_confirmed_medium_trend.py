import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2029-08-22')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); px={}
for s in syms:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=cutoff]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); r40=p.pct_change(40)
# Medium-horizon relative trend, confirmed by directional breadth and penalized by downside risk.
rel=r40.sub(r40.median(axis=1),axis=0)
bread=(r>0).rolling(20,min_periods=10).mean()
down=r.clip(upper=0).rolling(30,min_periods=15).std()*np.sqrt(252)
# rank-neutral confirmation: reward persistent positive paths, suppress unstable downside
confirm=(bread-0.5)*2
sig=(rel/(down+0.03))*(1+0.6*confirm)
sig=sig.shift(1)
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   vals.append(spearmanr(sig.loc[dt,ok],fwd.loc[dt,ok]).statistic); ns.append(ok.sum())
 vals=np.array(vals); print('H',h,'dates',len(vals),'IC %.6f ICIR %.6f avgN %.2f hit %.4f'%(np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals,ddof=1)*np.sqrt(252),np.mean(ns),np.mean(vals>0)))
 if h==20: 
  for name,mask in [('2020-25',sig.index<'2026-01-01'),('2026+',sig.index>='2026-01-01'),('2028+',sig.index>='2028-01-01'),('2029YTD',sig.index>='2029-01-01')]:
   x=[]
   for dt in sig.index[mask]:
    ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
    if ok.sum()>=8:x.append(spearmanr(sig.loc[dt,ok],fwd.loc[dt,ok]).statistic)
   x=np.array(x); print(name,'dates',len(x),'IC %.6f ICIR %.6f'%(np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1)*np.sqrt(252)))
ranks=sig.rank(axis=1,pct=True); print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),ranks.diff().abs().mean(axis=1).dropna().mean()))
out=sig; out.index.name='date'; out.to_csv('scripts/miner_2_20290823_breadth_confirmed_medium_trend_signal.csv')
