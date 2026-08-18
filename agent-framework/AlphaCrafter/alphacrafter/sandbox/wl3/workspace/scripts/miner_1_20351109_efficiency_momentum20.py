import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2035-11-07'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change()
rows=[]
for mom,pathw in [(10,20),(20,40),(40,80),(60,120)]:
 path=R.abs().rolling(pathw,min_periods=max(10,pathw//2)).sum()
 F=P.pct_change(mom).shift(1).div(path.shift(1)+1e-12)
 for i in range(max(pathw,mom)+5,len(P)-10):
  z=pd.concat([F.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(ic): rows.append((P.index[i],f'effmom_{mom}_{pathw}',ic,len(z)))
a=pd.DataFrame(rows,columns=['date','factor','ic','n'])
for f,g in a.groupby('factor'):
 x=g.set_index('date').ic; rec=x.iloc[-252:]
 print(f,'ic_dates',len(x),'avg_n',round(g.n.mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4),'recent',round(rec.mean(),6),round(rec.mean()/rec.std(),6),'coverage',round(g.n.mean()/15,4))
 if f=='effmom_20_40':
  F=P.pct_change(20).shift(1).div(R.abs().rolling(40,min_periods=20).sum().shift(1)+1e-12)
  F.to_csv('scripts/miner_1_20351109_efficiency_momentum20_signal.csv')
