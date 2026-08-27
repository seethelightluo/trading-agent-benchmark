import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
cl={}; vo={}
for s in U:
 d=pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date'); cl[s]=d.close.astype(float); vo[s]=d.volume.astype(float)
px=pd.DataFrame(cl).sort_index().loc[:'2035-06-20']; v=pd.DataFrame(vo).reindex(px.index); r=px.pct_change()
# Price-volume divergence: favor assets with negative 20D price impulse but weakening selling volume;
# risk scale by 60D volatility, lagged one session.
ret20=px/px.shift(20)-1
vol60=r.rolling(60,min_periods=30).std()
volratio=v.rolling(10,min_periods=5).mean()/(v.rolling(60,min_periods=30).mean()+1e-12)
# low recent volume after loss is contrarian confirmation; clip for robust cross-asset behavior
f=(-ret20/(vol60+0.005))*(2-volratio.clip(0,2))
f=f.replace([np.inf,-np.inf],np.nan).shift(1).clip(-10,10)
def calc(h):
 vals=[]; ds=[]; ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(px.index[i]); ns.append(len(z))
 x=pd.Series(vals,index=pd.DatetimeIndex(ds)); return x,len(x),np.mean(ns)
print('universe',len(U),'dates',px.index.min().date(),px.index.max().date())
for h in [5,10,20,40,60]:
 x,n,an=calc(h); print('H',h,'dates',n,'avgN',round(an,2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round((x>0).mean(),4))
x,_,_=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print('REG',a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6))
print('coverage',round(f.notna().sum(axis=1).div(len(U)).mean(),4),'turnover_proxy',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20350621_volume_divergence_reversal_signal.csv',index=False)
