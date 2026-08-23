import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=3000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Residual trend: asset 20d return relative to contemporaneous cross-sectional median,
# volatility scaled and confirmed by 20d breadth; lagged one completed session.
ret20=p/p.shift(20)-1
peer=ret20.median(axis=1)
res=ret20.sub(peer,axis=0)
vol=r.rolling(40).std()*np.sqrt(20)
breadth=ret20.gt(0).mean(axis=1)
confirm=(0.5+abs(breadth-0.5)*2) * (0.75+0.5*(breadth>0.5).astype(float))
fac=res.div(vol+1e-12).mul(confirm,axis=0).shift(1)
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
out=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('cutoff',out.index.max(),'dates',len(out),'avg_n',out.n.mean(),'assets',len(U),'coverage',fac.notna().sum().sum()/(len(fac)*len(U)))
if len(out):
 print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(out.ic.mean(),out.ic.mean()/(out.ic.std(ddof=1)+1e-12),(out.ic>0).mean(),fac.rank(axis=1).diff().abs().mean().mean()/len(U)))
 for name,sl in [('2020-22',out.loc['2020':'2022']),('2023-24',out.loc['2023':'2024']),('2025-27',out.loc['2025':'2027']),('online',out.loc['2026-07-16':'2027-04-07'])]: print(name,len(sl),sl.ic.mean() if len(sl) else np.nan,sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12) if len(sl)>1 else np.nan)
for h in [2,5,10,20]:
 yy=p.shift(-h)/p-1; rr=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr) if rr else np.nan,len(rr))
fac.to_csv('scripts/miner_1_20270408_residual_trend_signal.csv')
