import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; px={}
for s in U:
 f=f'{base}/{s}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close
p=pd.DataFrame(px).sort_index(); ix=p.index.intersection(v.index);p=p.loc[ix];v=v.loc[ix]
r=p.pct_change(20); high=(v>v.rolling(60).median()).astype(float)
# reversal in high volatility, trend in calm regime
f=r.multiply(2*high-1,axis=0); y=p.shift(-10)/p-1
I=[];D=[];N=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):I.append(q);D.append(dt);N.append(len(z))
I=np.array(I);print('dates',len(I),'avg_n',np.mean(N),'coverage',np.mean(N)/15);print('IC',I.mean(),'ICIR',I.mean()/I.std(ddof=1),'hit',np.mean(I>0))
for n in [120,260,520,780]:
 a=I[-n:];print('recent',n,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
os.makedirs('scripts/artifacts',exist_ok=True); rows=[]
for dt in D:
 for s in U:
  if s in p and pd.notna(f.loc[dt,s]): rows.append({'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':f.loc[dt,s]})
pd.DataFrame(rows).to_csv('scripts/artifacts/miner_1_20350201_vix_regime_reversal20_signal.csv',index=False)
