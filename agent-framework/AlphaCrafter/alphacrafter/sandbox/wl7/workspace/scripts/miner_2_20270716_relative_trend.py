import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
prices={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index(); prices[s]=d
rs={}; vs={}
for s,p in prices.items():
 r=p.pct_change(fill_method=None); rs[s]=p.pct_change(20,fill_method=None); vs[s]=r.rolling(20,min_periods=15).std()*np.sqrt(252)
r20=pd.DataFrame(rs); vol=pd.DataFrame(vs); factor=r20.sub(r20.median(axis=1),axis=0).div(vol).loc[:'2027-07-15']
# forward returns per asset in its own trading calendar, then align dates
fwd={}
for h in [1,5,10]:
 fwd[h]=pd.DataFrame({s:p.pct_change(h,fill_method=None).shift(-h) for s,p in prices.items()})
for h in [1,5,10]:
 ics=[]; ns=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(ics);print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
rank=factor.rank(axis=1,pct=True);print('coverage',round(factor.notna().sum().sum()/factor.size,4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'assets',len(prices))
vals=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],fwd[1].loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
x=pd.DataFrame(vals,columns=['date','ic']).set_index('date')
for label,a,b in [('2020-22','2020','2022'),('2023-24','2023','2024'),('2025-27','2025','2027')]:
 y=x.loc[a:b];print(label,len(y),round(y.ic.mean(),6),round(y.ic.mean()/y.ic.std(ddof=1),6))
sig=factor.stack().rename('signal').reset_index();sig.columns=['date','symbol','signal'];sig.to_csv('scripts/miner_2_20270716_relative_trend_signal.csv',index=False);print('artifact rows',len(sig))
