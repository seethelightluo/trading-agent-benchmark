import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].pct_change()
prices={}
for s in U:
 f=os.path.join(base,s+'.csv')
 if os.path.exists(f): prices[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close']
rets=pd.DataFrame(prices).pct_change(); macro=macro.reindex(rets.index)
rows=[]
for d in rets.index:
 hist=rets.loc[:d].tail(20); m=macro.loc[hist.index]
 if len(hist)<15 or m.notna().sum()<15: continue
 for s in U:
  x=hist[s]; z=pd.concat([x,m],axis=1).dropna()
  if len(z)<15 or z.iloc[:,1].var()==0: continue
  beta=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var()
  # defensive: lower DXY beta expected better next return
  nxt=rets[s].shift(-1).get(d,np.nan)
  if pd.notna(nxt): rows.append((d,s,-beta,nxt))
df=pd.DataFrame(rows,columns=['date','s','f','y'])
ics=df.groupby('date').apply(lambda x: spearmanr(x.f,x.y).statistic if len(x)>=8 and x.f.nunique()>1 else np.nan).dropna()
print('dates',len(ics),'avg names',df.groupby('date').size().mean(),'coverage',len(df)/(len(rets.index)*15),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(),'hit',(ics>0).mean())
for h in [5,10]:
 yy=rets.shift(-h)
 vals=[]
 for d,g in df.groupby('date'):
  z=[]
  for _,r in g.iterrows():
   v=yy.get(r.s,pd.Series()).get(d,np.nan)
   if pd.notna(v):z.append((r.f,v))
  if len(z)>=8: vals.append(spearmanr([a for a,b in z],[b for a,b in z]).statistic)
 vals=pd.Series(vals).dropna();print(h,len(vals),vals.mean(),vals.mean()/vals.std())
print('regimes')
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=ics[(ics.index>=a)&(ics.index<=b+'-12-31')];print(a,b,len(z),z.mean(),z.mean()/z.std())
# rank turnover
r=df.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True);print('turn',r.diff().abs().mean().mean())
print('last',df.date.max())
