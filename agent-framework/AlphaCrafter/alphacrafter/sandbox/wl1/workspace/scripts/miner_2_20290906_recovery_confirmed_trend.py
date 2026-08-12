import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2029-09-05')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); close={}
for s in syms:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 close[s]=d[d.index<=cutoff]
# Compute each asset on its native observations, then align cross-section.
sigparts={}
for s,p in close.items():
 r=p.pct_change(); ret60=p.pct_change(60)
 down=r.clip(upper=0).rolling(40,min_periods=20).std()*np.sqrt(252)
 recovery=r.rolling(10,min_periods=7).mean(); vol=r.rolling(40,min_periods=20).std()*np.sqrt(252)
 sigparts[s]=(ret60,down,recovery,vol)
idx=sorted(set().union(*[x.index for x in close.values()])); p=pd.DataFrame({s:close[s].reindex(idx) for s in syms},index=idx)
ret60=pd.DataFrame({s:x[0].reindex(idx) for s,x in sigparts.items()}); down=pd.DataFrame({s:x[1].reindex(idx) for s,x in sigparts.items()}); rec=pd.DataFrame({s:x[2].reindex(idx) for s,x in sigparts.items()}); vol=pd.DataFrame({s:x[3].reindex(idx) for s,x in sigparts.items()})
rel=ret60.sub(ret60.median(axis=1),axis=0)
confirm=np.tanh(rec/((vol/np.sqrt(252))+1e-6)*8)
sig=(rel/(down+0.025))*(1+0.45*confirm); sig=sig.replace([np.inf,-np.inf],np.nan).shift(1)
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(sig.loc[dt,ok],fwd.loc[dt,ok]).statistic); ns.append(ok.sum())
 vals=np.asarray(vals); ic=np.nanmean(vals); ir=ic/np.nanstd(vals,ddof=1)*np.sqrt(252)
 print(f'H {h} dates {len(vals)} IC {ic:.6f} ICIR {ir:.6f} avgN {np.mean(ns):.2f} hit {np.mean(vals>0):.4f}')
 if h==20:
  for name,mask in [('2020-25',sig.index<'2026-01-01'),('2026+',sig.index>='2026-01-01'),('2028+',sig.index>='2028-01-01'),('2029YTD',sig.index>='2029-01-01')]:
   x=[]
   for dt in sig.index[mask]:
    ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
    if ok.sum()>=8:x.append(spearmanr(sig.loc[dt,ok],fwd.loc[dt,ok]).statistic)
   x=np.asarray(x); print(name,'dates',len(x),'IC %.6f ICIR %.6f'%(np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1)*np.sqrt(252)))
ranks=sig.rank(axis=1,pct=True); print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),ranks.diff().abs().mean(axis=1).dropna().mean()))
sig.index.name='date';sig.to_csv('scripts/miner_2_20290906_recovery_confirmed_trend_signal.csv')
