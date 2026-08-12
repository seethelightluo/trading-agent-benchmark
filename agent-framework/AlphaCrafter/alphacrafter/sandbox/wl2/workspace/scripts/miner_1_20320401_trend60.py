import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); r60=p.pct_change(60); v=r.rolling(20).std()*np.sqrt(252)
f=r60/v.replace(0,np.nan); agree=p.pct_change(20).gt(0).mean(axis=1)>=.60; f=f.where(agree); fw=p.shift(-1).div(p)-1
A=[];D=[];N=[]
for d in f.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);D.append(d);N.append(len(z))
A=np.array(A);D=pd.DatetimeIndex(D)
def s(a):return (round(a.mean(),6),round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),6),len(a)) if len(a)>1 else ('nan','nan',len(a))
print('candidate=60d_momentum_vol20_agreement_gate');print('dates',len(A),'avg_n',round(np.mean(N),3),'coverage',round(f.notna().sum().sum()/f.size,4),'active',round(agree.mean(),4),'IC ICIR n',s(A),'hit',round(np.mean(A>0),4))
for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2029-12-31'),('2030','2032-04-01')]:print(lo,s(A[(D>=pd.Timestamp(lo))&(D<=pd.Timestamp(hi))]))
for h in [3,5,10]:
 fw=p.shift(-h).div(p)-1;a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,s(np.array(a)))
f.insert(0,'date',f.index);f.to_csv('scripts/miner_1_20320401_trend60_signal.csv',index=False)
