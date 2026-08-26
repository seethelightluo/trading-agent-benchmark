import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end='2034-12-11'
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:end]
 px[s]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# directional efficiency: signed 20d return divided by sum absolute daily returns, lagged by one date
mom=p.pct_change(20)
eff=mom/(r.abs().rolling(20).sum()+1e-12)
# volatility damped efficiency, cross-sectional demeaned to avoid common market direction
vol=r.rolling(20).std()*np.sqrt(252)
f=-eff/(vol+1e-12)
f=f.shift(1)
rows=[]
for h in [1,5,10,20]:
 ic=[]; ns=[]; turnovers=[]
 fr=p.shift(-h)/p-1
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 # factor turnover based rank changes at successive valid dates
 for a,b in zip(f.index[:-1],f.index[1:]):
  xa=f.loc[a].rank(pct=True); xb=f.loc[b].rank(pct=True)
  q=pd.concat([xa,xb],axis=1).dropna()
  if len(q)>=8: turnovers.append(np.mean(np.abs(q.iloc[:,0]-q.iloc[:,1])))
 ic=np.array(ic); mean=ic.mean(); sd=ic.std(ddof=1); icir=mean/sd if sd else np.nan
 print(f'H{h}: dates={len(ic)} avgN={np.mean(ns):.1f} IC={mean:.6f} ICIR={icir:.6f} hit={np.mean(ic>0):.3f} turnover={np.mean(turnovers):.4f}')
print('coverage', f.notna().sum(axis=1).mean()/15, 'last', f.dropna().tail(1).index[-1])
# recent
for h in [10]:
 fr=p.shift(-h)/p-1; a=[]
 for dt in f.index[-500:]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a); print('recent500',len(a),a.mean(),a.mean()/a.std(ddof=1))
# artifact
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); out.to_csv('scripts/miner_2_20341211_efficiency_signal.csv',index=False)
