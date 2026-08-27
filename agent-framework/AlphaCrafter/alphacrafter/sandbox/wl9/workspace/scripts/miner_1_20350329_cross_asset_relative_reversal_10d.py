import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{root}/{s}.csv'); d['date']=pd.to_datetime(d['date']); px[s]=d.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=np.log(p).diff(); ret10=r.rolling(10).sum(); rel=ret10.sub(ret10.median(axis=1),axis=0); vol=r.rolling(20).std(); sig=(-rel/vol).shift(1); fwd=np.log(p.shift(-10))-np.log(p)
rows=[]; dates=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt)
a=np.array(rows); print('dates',len(a),'avgN',np.mean([pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna().shape[0] for d in dates]),'IC',a.mean(),'ICIR',a.mean()/a.std()*np.sqrt(252),'hit',np.mean(a>0))
for lo,hi in [(2024,2026),(2027,2029),(2030,2032),(2033,2036)]:
 q=np.array([v for d,v in zip(dates,a) if lo<=d.year<=hi]); print(lo,hi,len(q),q.mean(),q.mean()/q.std()*np.sqrt(252))
rr=sig.rank(axis=1,pct=True); tv=[]
for d1,d2 in zip(rr.index[:-1],rr.index[1:]):
 z=pd.concat([rr.loc[d1],rr.loc[d2]],axis=1).dropna();
 if len(z)>=8: tv.append(abs(z.iloc[:,0]-z.iloc[:,1]).mean())
print('coverage',sig.notna().mean().mean(),'turnover',np.mean(tv))
out=sig.loc[dates].copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20350329_cross_asset_relative_reversal_10d_signal.csv')
