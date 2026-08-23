import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
acct=get_account_dict(); universe=acct.get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in universe:
 d=get_stock_daily_data(s,2000)
 if d is None or len(d)<150: d=get_index_daily_data(s,2000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); frames[s]=d.set_index('date')['close'].astype(float).sort_index()
px=pd.DataFrame(frames).sort_index(); ret=np.log(px).diff()
r20=px.pct_change(20); r40=px.pct_change(40).shift(20); vol=ret.rolling(60).std()*np.sqrt(252)
sig=((r20-r40)/vol).shift(1); fwd=px.pct_change(1).shift(-1)
ics=[]; dates=[]; turnovers=[]; nobs=[]; covs=[]
for i,dt in enumerate(sig.index):
 a=sig.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  ics.append(a[ok].corr(b[ok],method='spearman')); dates.append(dt); nobs.append(ok.sum()); covs.append(ok.mean())
  if i: 
   prev=sig.iloc[i-1]; turnovers.append((a[ok].rank(pct=True)-prev[ok].rank(pct=True)).abs().mean())
ics=pd.Series(ics,index=pd.to_datetime(dates)).dropna()
print('candidate curvature_20v40 | dates',len(ics),'avg_n',round(np.mean(nobs),3),'coverage',round(np.mean(covs),4))
print('IC',round(ics.mean(),6),'ICIR',round(ics.mean()/ics.std(ddof=1),6),'hit',round((ics>0).mean(),4),'turnover',round(np.mean(turnovers),4))
for h in [5,10,20]:
 ff=px.pct_change(h).shift(-h); z=[]
 for dt in sig.index:
  a=sig.loc[dt]; b=ff.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: z.append(a[ok].corr(b[ok],method='spearman'))
 print('decay',h,round(pd.Series(z).dropna().mean(),6))
for name,mask in [('2020-22',ics.index<'2023-01-01'),('2023-24',(ics.index>='2023-01-01')&(ics.index<'2025-01-01')),('2025+',ics.index>='2025-01-01')]:
 x=ics[mask]; print('regime',name,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20270421_curvature_20v40_signal.csv',index=False)
