import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: x=fn(s,days=5000)
  except Exception: x=None
  if x is not None and len(x): break
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date']); vix=vix.set_index('date')['close'].reindex(p.index).ffill()
# Regime-conditioned momentum: 20d residual return, direction flips to reversal in high-VIX regime; lag all inputs.
ret20=p/p.shift(20)-1; resid=ret20-ret20.median(axis=1).values[:,None]
vol=r.rolling(30,min_periods=20).std(); base=resid/vol
threshold=vix.rolling(252,min_periods=126).quantile(.75)
high=(vix>threshold).astype(float)
f=(base*(1-2*high)).shift(1)
rows=[]
for i in range(len(p)-11):
 z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+11]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],z.f.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('shape',p.shape,'dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15),'last',p.index[-1].date())
print('H10 IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=a.loc[lo:hi]
 if len(q): print(lo+'-'+hi,len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('recent120',a.tail(120).ic.mean(),a.tail(120).ic.mean()/a.tail(120).ic.std(ddof=1))
print('turnover',(f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean())
f.index.name='date'; f.reset_index().to_csv('scripts/miner_1_20310918_vix_regime_momentum_signal.csv',index=False)
