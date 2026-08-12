import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-12-24'); px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close.reindex(pd.date_range('2020-01-01',end)).ffill()
p=pd.DataFrame(px); r=np.log(p).diff(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill()
# shock reversal: recent 5d loss, scaled by own vol, activated by a lagged VIX shock
shock=(v.pct_change(5)>v.pct_change(5).rolling(60).median()).astype(float)
f=-r.rolling(5).sum()/(r.rolling(20).std()+1e-8)*shock
for h in [1,5,10,20]:
 a=[]; ns=[]
 for i in range(25,len(p)-h-1):
  z=pd.concat([f.iloc[i-1],np.log(p.iloc[i+h]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a); print(h,len(a),np.mean(ns),np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),np.mean(a>0))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for a0,b in [('2020-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-24')]:
 q=[]
 for i in range(25,len(p)-20-1):
  if a0<=str(p.index[i].date())<=b:
   z=pd.concat([f.iloc[i-1],np.log(p.iloc[i+20]/p.iloc[i])],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('regime',a0,len(q),np.nanmean(q),np.nanmean(q)/(np.nanstd(q,ddof=1)+1e-12))
f.to_csv('scripts/miner_2_20311225_vix_shock_reversal_signal.csv')
