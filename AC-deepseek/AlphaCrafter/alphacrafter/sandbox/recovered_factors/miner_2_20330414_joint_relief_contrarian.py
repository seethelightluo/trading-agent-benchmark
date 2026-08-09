import pandas as pd,numpy as np
from scipy.stats import spearmanr
W=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in W}).sort_index().loc[:'2033-04-10'];r=P.pct_change();v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill();d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill()
dv=v.pct_change();dd=d.pct_change();zv=(dv-dv.rolling(60,min_periods=30).mean())/(dv.rolling(60,min_periods=30).std()+1e-12);zd=(dd-dd.rolling(60,min_periods=30).mean())/(dd.rolling(60,min_periods=30).std()+1e-12)
relief=(-zv.clip(-3,3)-zd.clip(-3,3)).clip(0,6)
# Relief creates crowded rebound; contrarian version fades recent momentum following joint relief.
F=-(r.rolling(10,min_periods=8).sum()*(1+relief).values[:,None]/(r.rolling(20,min_periods=15).std()+1e-12)).shift(1)
print('candidate=joint_relief_contrarian_10obs; dates',len(P),'assets',len(W),'coverage',round(F.notna().mean().mean(),4),'meanN',round(F.notna().sum(axis=1).replace(0,np.nan).mean(),2),'turn10',round(F.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;a=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lab,lo,hi in [('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-04-10')]:
 fw=P.shift(-1)/P-1;a=[]
 for dt in F.loc[lo:hi].index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print('REG1',lab,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('AUDIT_REQUIRED')
