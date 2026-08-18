import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3200); D[s]=x.set_index(pd.to_datetime(x.date)).close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change();
# Momentum rewarded only when it is not dominated by downside risk: 20d return / 40d downside deviation.
down=r.where(r<0,0).rolling(40,min_periods=30).std(); mom=p.pct_change(20); f=(mom/(down*np.sqrt(252)+1e-8)).replace([np.inf,-np.inf],np.nan).shift(1); fr=p.shift(-10)/p-1
ics=[]; ns=[]; dates=[]
for t in f.index:
 z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(q): ics.append(q);ns.append(len(z));dates.append(t)
x=pd.Series(ics,index=dates); print('dates',len(x),'avgN',round(np.mean(ns),2),'instruments',len(U),'end',x.index.max().date())
def metrics(y): return np.mean(y),np.mean(y)/(np.std(y,ddof=1)/np.sqrt(len(y))) if len(y)>1 else np.nan
print('full IC ICIR hit',*(round(v,6) for v in (*metrics(x.values),np.mean(x>0))))
for n in [365,730,1095]:
 y=x[x.index>=x.index.max()-pd.Timedelta(days=n)];print('recent',n,'dates',len(y),'IC/ICIR',*(round(v,6) for v in metrics(y.values)))
print('coverage',round(f.notna().sum().sum()/p.notna().sum().sum(),4),'rank_turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
