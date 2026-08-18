import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-09-15')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
r=P.pct_change(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
res20=resid.rolling(20,min_periods=12).sum(); vol=resid.rolling(60,min_periods=30).std()*np.sqrt(252)
disp=resid.std(axis=1).rolling(20,min_periods=15).mean()
scale=(disp/(disp.rolling(120,min_periods=60).median()+1e-12)).clip(0.5,2.0)
sig=(-(res20/vol.replace(0,np.nan)).mul(scale,axis=0)).shift(1)
rows=[]
for h in [10,20,30]:
 y=P.shift(-h)/P-1; vals=[]; ns=[]; turns=[]; prev=None
 for d in sig.index:
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
   if prev is not None:
    a=sig.loc[d].reindex(U); b=prev.reindex(U)
    turns.append(np.nanmean(np.abs((a-a.mean())/(a.abs().mean()+1e-12)-(b-b.mean())/(b.abs().mean()+1e-12))))
   prev=sig.loc[d]
 x=np.asarray(vals); ic=np.nanmean(x); icir=ic/(np.nanstd(x,ddof=1)+1e-12)*np.sqrt(252)
 print(f'H{h} dates={len(x)} avgN={np.mean(ns):.2f} IC={ic:.8f} ICIR={icir:.8f} hit={np.mean(x>0):.4f} turnover={np.nanmean(turns):.6f}')
 for label,lo,hi in [('early','2020-01-01','2026-01-01'),('mid','2026-01-01','2030-01-01'),('recent','2030-01-01','2032-09-16')]:
  z=[]
  for d in sig.index[(sig.index>=pd.Timestamp(lo))&(sig.index<pd.Timestamp(hi))]:
   q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
   if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
  print(' ',label,'dates',len(z),'IC',round(float(np.nanmean(z)),6) if z else None)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_1_20320916_dispersion_gated_residual_reversal_signal.csv',index=False)
print('artifact_rows',len(out),'coverage',round(float(sig.notna().mean().mean()),6),'assets',len(U),'cut',CUT.date())
