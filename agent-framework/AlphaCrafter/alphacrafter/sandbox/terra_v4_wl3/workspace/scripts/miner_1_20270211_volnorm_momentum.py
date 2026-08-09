import numpy as np, pandas as pd
UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in UNIV:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index(); R=P.pct_change(); ret=P/P.shift(20)-1; vol=R.rolling(20).std()*np.sqrt(20); F=ret/vol.replace(0,np.nan)
ics=[]; turnovers=[]; counts=[]; regime=[]
for i in range(20,len(P)-1):
 f=F.iloc[i]; y=R.iloc[i+1]; z=pd.concat([f,y],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); ics.append(c); counts.append(len(z)); old=F.iloc[i-1].reindex(z.index); turnovers.append(np.mean(np.sign(z.iloc[:,0])!=np.sign(old.fillna(0)))); regime.append((P.index[i],c))
a=np.array(ics); a=a[np.isfinite(a)]
print('candidate=vol_normalized_20d_momentum dates',len(a),'avg_names',round(float(np.mean(counts)),2),'coverage',round(float(np.sum(counts)/(len(a)*15)),4))
print('IC',round(float(a.mean()),6),'ICIR',round(float(a.mean()/a.std()),6),'hit',round(float(np.mean(a>0)),4),'turnover',round(float(np.nanmean(turnovers)),4))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=np.array([v for d,v in regime if lo<=str(d)[:4]<=hi]); print(lo+'-'+hi,'n',len(q),'ic',round(float(np.nanmean(q)),6))
print('decay')
for h in [1,5,10]:
 z=[]
 for i in range(20,len(P)-h):
  q=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print(h,len(z),round(float(np.nanmean(z)),6),round(float(np.nanmean(z)/np.nanstd(z)),6))
