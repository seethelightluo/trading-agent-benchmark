import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; V={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
  P[a]=d['close'].astype(float); V[a]=d['volume'].astype(float) if 'volume' in d else pd.Series(index=d.index,dtype=float)
P=pd.DataFrame(P).sort_index().loc[:'2033-12-29']; V=pd.DataFrame(V).reindex(P.index)
R=P.pct_change(); vol20=R.rolling(20,min_periods=15).std()
# Contrarian return, emphasized when current volume is unusually high; all inputs lagged one day.
shock=np.log((V.rolling(5,min_periods=3).mean()+1e-12)/(V.rolling(60,min_periods=30).median()+1e-12))
F=(-P.pct_change(10)*shock).shift(1)
FR=P.pct_change(10).shift(-10)
ics=[]; nobs=[]; turnovers=[]
for dt in F.index:
 x=F.loc[dt]; y=FR.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): ics.append(q); nobs.append(len(z))
# rank turnover on consecutive valid cross-sections
last=None
for dt in F.index:
 x=F.loc[dt].dropna()
 if len(x)>=8:
  r=x.rank(pct=True)
  if last is not None:
   common=r.index.intersection(last.index)
   if len(common)>=8: turnovers.append(np.mean(np.abs(r[common]-last[common])))
  last=r
ics=np.array(ics); print('candidate=volume_shock_reversal_10d dates=%d avg_instruments=%.2f IC=%.6f ICIR=%.6f hit=%.4f coverage=%.4f turnover=%.4f universe=%d'%(len(ics),np.mean(nobs),np.mean(ics),np.mean(ics)/np.std(ics,ddof=1),np.mean(ics>0),F.notna().sum().sum()/(F.shape[0]*len(A)),np.mean(turnovers),len(A)))
for days in [120,260,520,780]:
 z=ics[-days:] if len(ics)>=days else ics
 print('recent',days,'IC=%.6f ICIR=%.6f'%(np.mean(z),np.mean(z)/np.std(z,ddof=1)))
# artifacts for provenance
os.makedirs('scripts/artifacts',exist_ok=True)
pd.DataFrame({'date':F.index,'signal_mean':F.mean(axis=1)}).to_csv('scripts/artifacts/miner_1_20340105_volume_shock_reversal_10d_signal.csv',index=False)
pd.DataFrame({'ic':ics}).to_csv('scripts/artifacts/miner_1_20340105_volume_shock_reversal_10d_ic.csv',index=False)
