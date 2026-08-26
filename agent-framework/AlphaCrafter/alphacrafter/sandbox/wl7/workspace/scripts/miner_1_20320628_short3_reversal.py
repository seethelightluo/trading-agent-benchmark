import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2032-06-27')
files=glob.glob('../persistent/stock_data/*.csv')
data={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in files}
px=pd.concat(data,axis=1).sort_index().loc[:cutoff]
ret=px.pct_change()
# Lag-safe short-term reversal: inverse 3-session return, scaled by trailing 20d volatility
factor=-(px.shift(3)/px.shift(6)-1)/(ret.shift(3).rolling(20,min_periods=12).std()*np.sqrt(3))
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; ics=[]; ns=[]; dates=[]
 for d in factor.index:
  z=pd.concat([factor.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(ic): ics.append(ic);ns.append(len(z));dates.append(d)
 a=np.array(ics)
 print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={(a>0).mean():.3f}')
 if h==10:
  ranks=factor.rank(axis=1,pct=True); valid=ranks.notna().sum(axis=1)>=8
  rr=ranks[valid].diff().abs().mean(axis=1).dropna()
  print(f'coverage={factor.notna().sum(axis=1).mean()/15:.4f} turnover={rr.mean():.4f} first={dates[0].date()} last={dates[-1].date()}')
  factor.loc[dates].rename_axis('date').to_csv('scripts/miner_1_20320628_short3_reversal_signal.csv')
  for i,part in enumerate(np.array_split(a,3),1): print(f'H10 third{i} IC={part.mean():.6f} ICIR={part.mean()/part.std(ddof=1):.6f}')
