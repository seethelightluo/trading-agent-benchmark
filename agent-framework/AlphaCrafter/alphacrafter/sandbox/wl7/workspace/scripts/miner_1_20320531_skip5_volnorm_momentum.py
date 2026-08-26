import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2032-05-30')
files=glob.glob('../persistent/stock_data/*.csv')
data={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in files}
px=pd.concat(data,axis=1).sort_index().loc[:cutoff]
ret=px.pct_change()
factor=px.shift(5)/px.shift(25)-1
vol=ret.shift(5).rolling(40,min_periods=25).std()*np.sqrt(20)
factor=factor/vol.replace(0,np.nan)
for h in [5,10,20]:
 fwd=px.shift(-h)/px-1; ics=[]; ns=[]; dates=[]
 for d in factor.index:
  z=pd.concat([factor.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(ic): ics.append(ic);ns.append(len(z));dates.append(d)
 a=np.array(ics); print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={(a>0).mean():.3f}')
 if h==10:
  ranks=factor.rank(axis=1,pct=True); valid=ranks.notna().sum(axis=1)>=8
  rr=ranks[valid].diff().abs().mean(axis=1).dropna()
  print(f'coverage={factor.notna().sum(axis=1).mean()/15:.4f} turnover={rr.mean():.4f} first={dates[0].date()} last={dates[-1].date()}')
  factor.loc[dates].rename_axis('date').to_csv('scripts/miner_1_20320531_skip5_volnorm_momentum_signal.csv')
