import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s,n=2300):
 d=get_stock_daily_data(s,n)
 if d is None or len(d)<100: d=get_index_daily_data(s,n)
 return d
px={s:get(s) for s in U}; rows=[]
for s,d in px.items():
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date'); r=d.close.pct_change()
 # Efficient medium-term trends: signed 40d return, scaled by realized risk and path efficiency.
 vol=r.rolling(20).std(); net=d.close.pct_change(40).abs(); path=r.abs().rolling(40).sum(); eff=(net/path).replace([np.inf,-np.inf],np.nan)
 f=d.close.pct_change(40)/vol.replace(0,np.nan)*eff
 for i in range(65,len(d)-10): rows.append((d.date.iloc[i],s,f.iloc[i],d.close.iloc[i+10]/d.close.iloc[i]-1))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna()
def daily(z):
 out=[]
 for _,g in z.groupby('date'):
  if len(g)>=8: out.append(g.factor.corr(g.fwd,method='spearman'))
 return pd.Series(out).dropna()
def stats(z):
 ic=daily(z); return len(ic),z.groupby('date').size().mean(),ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean(),np.mean([n/15 for n in z.groupby('date').size()])
print('overall',stats(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]: print('regime',a,b,stats(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 rr=[]
 for s,d in px.items():
  if d is None: continue
  d=d.copy().sort_values('date'); r=d.close.pct_change(); vol=r.rolling(20).std(); eff=d.close.pct_change(40).abs()/r.abs().rolling(40).sum(); f=d.close.pct_change(40)/vol.replace(0,np.nan)*eff
  for i in range(65,len(d)-h): rr.append((d.date.iloc[i],s,f.iloc[i],d.close.iloc[i+h]/d.close.iloc[i]-1))
 z=pd.DataFrame(rr,columns=['date','symbol','factor','fwd']).dropna(); print('decay',h,stats(z)[0:4])
# rank turnover
p=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean(axis=1).mean())
