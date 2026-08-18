import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 return d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close.astype(float)
P=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=P.pct_change()
# residual medium-term momentum: remove contemporaneous equal-weight world move
m=r.mean(axis=1,skipna=True)
res=r.sub(m,axis=0)
raw=res.rolling(60,min_periods=40).sum()
vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
f=(raw/vol.replace(0,np.nan)).shift(1)
# cross-sectional rank preserves comparability and lowers scale sensitivity
f=f.rank(axis=1,pct=True)
f.to_csv('../persistent/miner_2_20350413_residual_momentum60_signal.csv',index_label='date')
for h in [5,10,20,40]:
 out=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: out.append((P.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
 ic=q.ic.mean(); sd=q.ic.std(); ir=ic/sd*np.sqrt(252) if sd else np.nan
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((q.ic>0).mean(),4))
 for name,mask in [('pre2025',q.index<'2025-01-01'),('2025-2029',(q.index>='2025-01-01')&(q.index<'2030-01-01')),('2030+',q.index>='2030-01-01')]:
  z=q.loc[mask]; print(name,'IC',round(z.ic.mean(),6),'IR',round(z.ic.mean()/z.ic.std()*np.sqrt(252),6) if len(z)>1 else np.nan,'n',len(z))
print('universe',len(U),'rows',len(P),'coverage',round(f.notna().mean().mean(),4),'rank_turnover',round((f.diff().abs().mean(axis=1).mean()),6))
