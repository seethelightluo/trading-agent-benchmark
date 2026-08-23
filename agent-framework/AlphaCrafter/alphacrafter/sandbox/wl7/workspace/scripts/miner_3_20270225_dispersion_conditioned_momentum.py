import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d): prices[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(prices).sort_index(); r=p.pct_change()
base=(p/p.shift(20)-1)/(r.rolling(60).std()*np.sqrt(20)+1e-12)
disp=r.rolling(20).std().mean(axis=1)
dstate=(disp/(disp.rolling(120).median()+1e-12)).clip(.5,1.5)
breadth=r.gt(0).rolling(20).mean().mean(axis=1)
fac=base.mul((.5+breadth)*dstate,axis=0).shift(1)
fwd=p.shift(-1)/p-1
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
out=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('cutoff',out.index.max(),'dates',len(out),'avg_n',out.n.mean(),'assets',len(U),'coverage',fac.notna().sum().sum()/(len(fac)*len(U)))
if len(out):
 print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(out.ic.mean(),out.ic.mean()/(out.ic.std(ddof=1)+1e-12),(out.ic>0).mean(),fac.rank(axis=1).diff().abs().mean().mean()/len(U)))
 for name,sl in [('2020-22',out.loc['2020':'2022']),('2023-24',out.loc['2023':'2024']),('2025-27',out.loc['2025':'2027']),('online',out.loc['2026-07-16':'2027-02-24'])]: print(name,len(sl),sl.ic.mean() if len(sl) else np.nan,sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12) if len(sl)>1 else np.nan)
for h in [2,5,10,20]:
 yy=p.shift(-h)/p-1; rr=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr) if rr else np.nan,len(rr))
fac.loc[:'2027-02-24'].to_csv('scripts/miner_3_20270225_dispersion_conditioned_momentum_signal.csv')
