import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-01-07')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut];px[s]=d
p=pd.concat(px,axis=1).sort_index();r=p.pct_change()
# Favor assets with positive skew / fewer downside days: downside deviation relative to total vol
neg=r.where(r<0,0); down=np.sqrt((neg**2).rolling(30).mean()); tot=r.rolling(30).std(); f=-(down/tot.replace(0,np.nan))
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print('H',h,'IC %.8f ICIR %.5f hit %.4f dates %d avgN %.3f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),len(a),np.mean(ns)))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
