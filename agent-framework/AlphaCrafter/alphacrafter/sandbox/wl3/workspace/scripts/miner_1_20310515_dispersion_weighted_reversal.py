import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except:pass
  if x is not None and len(x):break
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Continuous dispersion-weighted residual reversal: fade 3-session cross-sectional shocks,
# with stronger signal when cross-asset dispersion is elevated; volatility normalized.
r3=r.rolling(3).sum(); resid=r3.sub(r3.median(axis=1),axis=0)
disp=resid.abs().median(axis=1); gate=disp/(disp.rolling(252).median()+1e-12)
f=-resid/(r.rolling(20).std()*np.sqrt(3)+1e-12).mul(gate.clip(upper=3),axis=0)
def corr(a,b):
 z=pd.concat([a.rename('f'),b.rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 return z.f.corr(z.y) if len(z)>=8 and z.f.std()>0 and z.y.std()>0 else np.nan
def ev(h):
 out=[]
 for i in range(len(p)-h-1):out.append(corr(f.iloc[i],p.iloc[i+h]/p.iloc[i+1]-1))
 a=pd.Series(out,index=p.index[:len(out)]).dropna(); print('H',h,'dates',len(a),'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
  q=a.loc[lo:hi]
  if len(q):print(lo+'-'+hi,len(q),'IC %.8f ICIR %.8f'%(q.mean(),q.mean()/q.std(ddof=1)))
 return a
for h in [1,3,5,6,10]:ev(h)
print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'turnover',(f.rank(pct=True).diff().abs().mean(axis=1)/2).mean(),'last',p.index[-1].date())
f.index.name='date';f.reset_index().to_csv('scripts/miner_1_20310515_dispersion_weighted_reversal_signal.csv',index=False)
