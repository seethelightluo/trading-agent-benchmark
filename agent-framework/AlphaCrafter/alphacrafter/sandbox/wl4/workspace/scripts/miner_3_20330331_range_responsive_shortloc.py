import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; fs={}; ps={}
for a in assets:
 f=f'{base}/{a}.csv'
 if not os.path.exists(f): continue
 d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); c=d.close.astype(float); r=c.pct_change()
 mom=c.pct_change(60); vol=r.rolling(90,min_periods=45).std()*np.sqrt(60)
 loc=((c-d.low)/(d.high-d.low).replace(0,np.nan)).rolling(10,min_periods=5).mean()
 fs[a]=(mom/vol*loc).shift(1); ps[a]=c
F=pd.DataFrame(fs); P=pd.DataFrame(ps); rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],P.shift(-30).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('dates',len(r),'avgN',r.n.mean(),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
print('recent260',s.tail(260).mean(),s.tail(260).mean()/s.tail(260).std(),'recent520',s.tail(520).mean(),s.tail(520).mean()/s.tail(520).std(),'recent780',s.tail(780).mean(),s.tail(780).mean()/s.tail(780).std())
print('coverage',F.notna().sum(axis=1).mean()/len(assets),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
r.to_csv('scripts/artifacts/miner_3_20330331_range_responsive_shortloc_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_3_20330331_range_responsive_shortloc_signal.csv')
