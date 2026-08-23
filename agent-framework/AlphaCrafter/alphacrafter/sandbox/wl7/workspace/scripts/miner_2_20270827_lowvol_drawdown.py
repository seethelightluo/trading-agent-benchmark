import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
END=pd.Timestamp('2027-08-27'); px={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); px[s]=d.set_index('date')['close']
c=pd.DataFrame(px).sort_index().loc[:END]; r=c.pct_change()
# defensive low-volatility with drawdown penalty: low realized risk and shallow drawdown
vol=r.rolling(20,min_periods=15).std(); dd=c/c.rolling(60,min_periods=30).max()-1
f=-(vol)*(1+(-dd).clip(lower=0).rolling(10,min_periods=5).mean())
for h in [1,5,10]:
 a=[]; ns=[]
 for dt in c.index:
  z=pd.concat([f.loc[dt],c.shift(-h).loc[dt]/c.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a); print(h,len(a), 'IC',a.mean(),'sd',a.std(ddof=1),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'N',np.mean(ns))
print('dates',len(c),'coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 a=[]
 for dt in c.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],c.shift(-1).loc[dt]/c.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(lo,hi,len(a),np.mean(a) if a else None)
f.to_csv('scripts/miner_2_20270827_lowvol_drawdown_signal.csv')
