import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-03-24'); X={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end]
 rng=(d.high-d.low).replace(0,np.nan); clv=(2*d.close-d.high-d.low)/rng
 # pressure persistence, volume-independent to preserve coverage
 f=clv.rolling(3).mean(); r=d.close.pct_change(); X[s]=pd.DataFrame({'f':f,'r':r})
D=sorted(set().union(*[x.index for x in X.values()])); rows=[]; signals=[]
for dt in D:
 a=[]
 for s in U:
  if dt in X[s].index and pd.notna(X[s].loc[dt,'f']):
   ix=X[s].index.get_loc(dt)
   if ix+1<len(X[s]):
    z=X[s].iloc[ix+1].r
    if pd.notna(z): a.append((s,X[s].loc[dt,'f'],z)); signals.append((dt,s,X[s].loc[dt,'f']))
 if len(a)>=8: rows += [(dt,*q) for q in a]
df=pd.DataFrame(rows,columns=['date','symbol','f','fr']); ic=df.groupby('date').apply(lambda z:spearmanr(z.f,z.fr).statistic).dropna()
wide=pd.DataFrame(signals,columns=['date','symbol','f']).pivot(index='date',columns='symbol',values='f'); to=wide.rank(pct=True).diff().abs().mean(axis=1).mean()
print('dates',len(ic),'rows',len(df),'avgN',df.groupby('date').size().mean(),'coverage',len(df)/(len(ic)*15),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'turnover',to,'start',ic.index.min(),'end',ic.index.max())
for h in [5,10]:
 a=[]
 for dt in D:
  vals=[]
  for s in U:
   if dt in X[s].index and pd.notna(X[s].loc[dt,'f']):
    ix=X[s].index.get_loc(dt)
    if ix+h<len(X[s]): vals.append((X[s].loc[dt,'f'],X[s].iloc[ix+1:ix+h+1].r.add(1).prod()-1))
  if len(vals)>=8:a.append(spearmanr([x[0] for x in vals],[x[1] for x in vals]).statistic)
 a=pd.Series(a).dropna();print('h',h,len(a),a.mean(),a.mean()/a.std(ddof=1))
pd.DataFrame(signals,columns=['date','symbol','f']).to_csv('scripts/miner_1_20270325_clv3_signal.csv',index=False)
