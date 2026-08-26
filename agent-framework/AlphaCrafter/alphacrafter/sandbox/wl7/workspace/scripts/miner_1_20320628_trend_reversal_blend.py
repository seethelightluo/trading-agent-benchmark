import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2032-06-27')
files=glob.glob('../persistent/stock_data/*.csv')
data={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in files}
px=pd.concat(data,axis=1).sort_index().loc[:cutoff]; ret=px.pct_change()
# Trend-reversal blend: medium trend (60d) with recent 5d reversal, both lagged 1 day
vol=ret.shift(1).rolling(40,min_periods=25).std()
trend=px.shift(1)/px.shift(61)-1
rev=-(px.shift(1)/px.shift(6)-1)
factor=(trend/(vol*np.sqrt(60))).rank(axis=1,pct=True)+(rev/(vol*np.sqrt(5))).rank(axis=1,pct=True)
for h in [5,10,20]:
 fwd=px.shift(-h)/px-1; a=[]; ns=[]; ds=[]
 for d in factor.index:
  z=pd.concat([factor.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): a.append(q);ns.append(len(z));ds.append(d)
 a=np.array(a); print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={(a>0).mean():.3f}')
 if h==10:
  print(f'coverage={factor.notna().sum(axis=1).mean()/15:.4f} turnover={factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.4f} first={ds[0].date()} last={ds[-1].date()}')
  factor.loc[ds].rename_axis('date').to_csv('scripts/miner_1_20320628_trend_reversal_blend_signal.csv')
  for i,p in enumerate(np.array_split(a,3),1): print(f'H10 third{i} IC={p.mean():.6f} ICIR={p.mean()/p.std(ddof=1):.6f}')
