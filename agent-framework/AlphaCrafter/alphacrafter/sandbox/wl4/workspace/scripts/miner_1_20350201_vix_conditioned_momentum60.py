import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 f=f'{base}/{s}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date').close
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close
prices=pd.DataFrame(px).sort_index(); common=prices.index.intersection(v.index); prices=prices.loc[common]; v=v.loc[common]
mom=prices/prices.shift(60)-1
reg=(v > v.rolling(20).mean()).astype(float)
factor=mom.multiply(1-2*reg,axis=0); fwd=prices.shift(-10)/prices-1
ics=[]; dates=[]; nobs=[]
for dt in prices.index:
 z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  r=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(r): ics.append(r); dates.append(dt); nobs.append(len(z))
ics=np.array(ics); print('dates',len(ics),'avg_n',np.mean(nobs),'coverage',np.mean(nobs)/15)
print('IC',np.mean(ics),'ICIR',np.mean(ics)/np.std(ics,ddof=1),'hit',np.mean(ics>0))
for n in [120,260,520,780]:
 a=ics[-n:]; print('recent',n,'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
os.makedirs('scripts/artifacts',exist_ok=True); out=[]
for dt in dates:
 for s in U:
  if s in prices and pd.notna(factor.loc[dt,s]): out.append({'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':factor.loc[dt,s]})
pd.DataFrame(out).to_csv('scripts/artifacts/miner_1_20350201_vix_conditioned_momentum60_signal.csv',index=False)
