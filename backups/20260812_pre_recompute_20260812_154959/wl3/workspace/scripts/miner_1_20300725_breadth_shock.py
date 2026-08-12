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
# Candidate: broad-market breadth-conditioned 3d downside shock reversal.
r3=lr.rolling(3).sum(); med=r3.median(axis=1); shock=r3.sub(med,axis=0)
vol=lr.rolling(30).std()*np.sqrt(3); sig=(-shock/(vol+1e-12))
breadth=(lr.rolling(5).sum()<0).mean(axis=1)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(px.index).ffill()
vm=vix.rolling(90).median(); gate=((breadth>=.55)&(vix>vm)).astype(float)
f=sig.mul(gate,axis=0).shift(1).replace([np.inf,-np.inf],np.nan)
rows=[]
for h in [1,3,5,10]:
 q=[]
 for i,dt in enumerate(px.index[:-h]):
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(q,columns=['date','n','ic']).set_index('date'); rows.append(q)
 print('H',h,'obs',len(q),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(q.n.mean(),q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12),(q.ic>0).mean()))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  y=q.loc[a:b].ic
  if len(y): print(a+'-'+b,len(y),'IC %.6f ICIR %.6f'%(y.mean(),y.mean()/(y.std(ddof=1)+1e-12)))
print('dates',len(px),'instruments',len(D),'signal coverage %.4f active',f.notna().mean().mean(),int(gate.sum()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300725_breadth_shock_signal.csv',index=False)
