import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-11-07'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill();R=P.pct_change()
for kind in ['downside','upside']:
 den=(R.clip(upper=0) if kind=='downside' else R.clip(lower=0)).abs().rolling(40,min_periods=20).sum()
 F=P.pct_change(20).shift(1).div(den.shift(1)+1e-12); rows=[]
 for i in range(65,len(P)-10):
  z=pd.concat([F.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):rows.append((P.index[i],q,len(z)))
 x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').ic; rec=x.iloc[-252:]
 print(kind,'dates',len(x),'n',round(np.mean([r[2] for r in rows]),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4),'recent',round(rec.mean(),6),round(rec.mean()/rec.std(),6),'coverage',round(np.mean([r[2]/15 for r in rows]),4))
 if kind=='downside':F.to_csv('scripts/miner_1_20351109_downside_effmom20_signal.csv')
