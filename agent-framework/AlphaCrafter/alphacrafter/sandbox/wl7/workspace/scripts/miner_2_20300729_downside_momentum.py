import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=3000)
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').drop_duplicates('date');F[s]=d
px=pd.DataFrame({s:pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date) for s,d in F.items()}).sort_index().ffill()
r=np.log(px).diff(); ret10=np.log(px/px.shift(10)); downside=np.sqrt((r.where(r<0,0)**2).rolling(30,min_periods=15).mean())
# downside-risk-adjusted medium momentum, lagged one day; cross-sectional rank is used in IC
sig=(ret10/(downside*np.sqrt(10)+1e-8)).shift(1).clip(-10,10)
fwd={h:px.shift(-h)/px-1 for h in [1,5,10,20,40]}
def ev(h):
 out=[]
 for dt in px.index:
  z=pd.concat([sig.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
  if len(z)>=8:out.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 return pd.DataFrame(out,columns=['date','n','ic']).dropna()
for h in [1,5,10,20,40]:
 q=ev(h);m=q.ic.mean();ir=m/q.ic.std(ddof=1)*np.sqrt(252)
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC %.8f ICIR %.8f hit %.4f'%(m,ir,(q.ic>0).mean()))
 if h==10:
  for lab,ss in [('early',q.iloc[:len(q)//3]),('middle',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]:
   print(lab,'dates',len(ss),'IC %.8f ICIR %.8f'%(ss.ic.mean(),ss.ic.mean()/ss.ic.std(ddof=1)*np.sqrt(252)))
  q.to_csv('scripts/miner_2_20300729_downside_momentum_ic.csv',index=False)
rank=sig.rank(axis=1,pct=True)
print('assets',len(F),'dates',len(px),'coverage',round(sig.notna().sum().sum()/(len(px)*len(U)),6),'turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'asset'}).to_csv('scripts/miner_2_20300729_downside_momentum_signal.csv',index=False)
