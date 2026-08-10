import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in symbols}
px=pd.DataFrame(D).sort_index(); px=px.loc[:'2027-02-25']
# path-efficiency trend: directional 20d return divided by total absolute daily movement, stable trends rank high
ret=px.pct_change(); fac=(px/px.shift(20)-1)/(ret.abs().rolling(20).sum()+1e-12)
# same-day cross-sectional ranks not needed for IC
for h in [1,5,10]:
  fwd=px.shift(-h)/px-1
  ics=[]; turns=[]; nins=[]
  dates=fac.index
  for d in dates:
    a=fac.loc[d]; b=fwd.loc[d]; ok=a.notna()&b.notna()
    if ok.sum()>=8:
      ics.append(spearmanr(a[ok],b[ok]).statistic); nins.append(ok.sum())
      # turnover rank signal vs 10 sessions ago, on available names
      old=fac.shift(10).loc[d]; ko=ok&old.notna()
      if ko.sum()>=8: turns.append(np.mean(np.sign(a[ko])!=np.sign(old[ko])))
  x=np.array(ics); print('H',h,'dates',len(x),'avgN',np.mean(nins),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'turn',np.mean(turns) if turns else np.nan)
for p,(a,b) in enumerate([(2020,2022),(2023,2024),(2025,2026),(2027,2027)]):
 x=[]
 fwd=px.shift(-10)/px-1
 for d in fac.index:
  if d.year not in range(a,b+1):continue
  z=fac.loc[d]; y=fwd.loc[d]; ok=z.notna()&y.notna()
  if ok.sum()>=8:x.append(spearmanr(z[ok],y[ok]).statistic)
 print('regime',a,b,len(x),np.mean(x) if x else np.nan)
# artifact rows
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_path_efficiency20.csv',index=False)
print('coverage',fac.notna().mean().mean(),'matrix',fac.notna().sum().sum()/fac.size)
