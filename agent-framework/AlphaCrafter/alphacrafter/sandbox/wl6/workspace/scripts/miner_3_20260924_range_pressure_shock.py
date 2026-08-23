import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
series={}
for a in assets:
 p=f'{base}/{a}.csv'
 if not os.path.exists(p): continue
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 # daily range pressure: negative CLV times range normalized by ATR20
 tr=(d.high-d.low).abs()
 atr=tr.rolling(20,min_periods=10).mean()
 clv=2*(d.close-d.low)/(d.high-d.low).replace(0,np.nan)-1
 fac= -clv*tr/atr
 # next valid observation return
 fwd=d.close.shift(-1)/d.close-1
 series[a]=pd.DataFrame({'f':fac,'r':fwd})
rows=[]
for dt in sorted(set().union(*[set(x.index) for x in series.values()])):
 vals=[]
 for a,x in series.items():
  if dt in x.index and pd.notna(x.loc[dt,'f']) and pd.notna(x.loc[dt,'r']): vals.append((a,x.loc[dt,'f'],x.loc[dt,'r']))
 if len(vals)>=8:
  rows.append([dt,spearmanr([z[1] for z in vals],[z[2] for z in vals]).statistic,len(vals)])
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.sum()/(len(z)*15))
for h in [1,5,10]:
 rr=[]
 for a,x in series.items():
  # forward h return from current close
  q=x.copy(); raw=pd.read_csv(f'{base}/{a}.csv'); raw.date=pd.to_datetime(raw.date); raw=raw.sort_values('date').set_index('date'); q['r']=raw.close.shift(-h)/raw.close-1
  for dt,v in q.dropna().iterrows(): rr.append((dt,a,v.f,v.r))
 dd=pd.DataFrame(rr,columns=['date','a','f','r']); out=[]
 for dt,g in dd.groupby('date'):
  if len(g)>=8: out.append(spearmanr(g.f,g.r).statistic)
 out=np.array(out); print(h,'IC',out.mean(),'ICIR',out.mean()/out.std(ddof=1),'hit',np.mean(out>0),'obs',len(out))
print('turnover proxy',np.nan)
