import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in symbols:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 frames[s]=d['close'].rename(s)
p=pd.concat(frames.values(),axis=1).sort_index()
r=p.pct_change()
# candidate: 60d trend normalized by trailing 20d risk, cross-section comparable, lag one day
mom=p.pct_change(60)
vol=r.rolling(20,min_periods=15).std()
sig=(mom/vol.replace(0,np.nan)).shift(1)
fwd=p.shift(-10)/p-1
rows=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# only dates with forward endpoint available and normal history
print('candidate=volscaled_momentum_60d')
print('dates',len(a),'avg_n',a.n.mean(),'coverage_panel',a.n.sum()/(len(a)*15))
print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1), (a.ic>0).mean()))
for days in [120,260,520,1040]:
 q=a.tail(days); print('recent',days,'IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
# rank turnover proxy
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna(); print('turnover_proxy',turn.mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr),np.nanmean(rr)/np.nanstd(rr,ddof=1))
os.makedirs('scripts/artifacts',exist_ok=True)
sig.to_csv('scripts/artifacts/miner_2_20350315_volscaled_momentum_60d_signal.csv')
a.to_csv('scripts/artifacts/miner_2_20350315_volscaled_momentum_60d_ic.csv')
