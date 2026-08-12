import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None and len(d): D[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); lr=np.log(px).diff()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(px.index).ffill()
# Downside-shock reversal: recent loss relative to cross-sectional median, scaled only by downside volatility.
r5=lr.rolling(5).sum(); csmed=r5.median(axis=1); shock=r5.sub(csmed,axis=0)
down=lr.where(lr<0).rolling(20).std(); down=down.fillna(lr.rolling(20).std())
# Macro gate uses only prior completed VIX observations and expands coverage in stressed regimes.
gate=(vix>vix.rolling(60).mean()+0.15*vix.rolling(60).std()).astype(float)
f=(-shock/(down+1e-12)).mul(gate,axis=0).replace([np.inf,-np.inf],np.nan).shift(1)
for h in [1,3,5,10]:
 q=[]
 for i,dt in enumerate(px.index[:-h]):
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(q,columns=['date','n','ic']).set_index('date')
 print('H',h,'obs',len(q),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(q.n.mean(),q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12),(q.ic>0).mean()))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  y=q.loc[a:b].ic
  if len(y): print(a+'-'+b,len(y),'%.6f %.6f'%(y.mean(),y.mean()/(y.std(ddof=1)+1e-12)))
print('dates',len(px),'instruments',len(D),'coverage %.4f active_days %d'%(f.notna().mean().mean(),int(gate.sum())))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20300711_downshock_reversal_signal.csv',index=False)
