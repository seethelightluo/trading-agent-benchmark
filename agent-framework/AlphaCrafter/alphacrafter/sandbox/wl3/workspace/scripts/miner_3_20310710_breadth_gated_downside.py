import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except Exception:x=None
  if x is not None and len(x): break
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); mom=p.pct_change(20); down=r.clip(upper=0).rolling(40,min_periods=25).std(); base=mom/(down*np.sqrt(20)+1e-12)
# Broad risk breadth: fraction of assets with positive trailing 20d return.
breadth=(mom>0).mean(axis=1)
# Contrarian only in weak breadth; trend-following otherwise. Lag all information one day.
f=base.where(breadth>=0.40,-base).shift(1)
rows=[]
for i in range(len(p)-11):
 z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+11]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],z.f.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('shape',p.shape,'dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15),'last',p.index[-1].date())
print('H10 IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/(a.ic.std(ddof=1)+1e-12),(a.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=a.loc[lo:hi]
 if len(q): print(lo+'-'+hi,len(q),q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12))
for h in [5,10,20]:
 rr=[]
 for i in range(len(p)-h-1):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h+1]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.corr(z.y))
 print('H',h,'IC',np.nanmean(rr),'ICIR',np.nanmean(rr)/(np.nanstd(rr,ddof=1)+1e-12))
print('turnover',(f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean()); f.index.name='date'; f.reset_index().to_csv('scripts/miner_3_20310710_breadth_gated_downside_signal.csv',index=False)
