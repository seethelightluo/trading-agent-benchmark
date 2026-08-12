import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-12-24')
px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d['close'].reindex(pd.date_range('2020-01-01',end,freq='D')).ffill()
p=pd.DataFrame(px).loc[:end]
r=np.log(p).diff()
# downside-adjusted medium trend: return / downside deviation, penalizing harmful volatility
ret=r.rolling(20).sum(); down=r.where(r<0,0).rolling(40).std(); sig=r.rolling(40).std()
f=ret/(down+0.5*sig+1e-8)
# lag to ensure only completed prior day; forward return from t to t+h
rows=[]
for h in [1,5,10,20]:
  ics=[]; nins=[]
  for i in range(40,len(p)-h-1):
   date=p.index[i]
   x=f.iloc[i-1]; y=np.log(p.iloc[i+h]/p.iloc[i])
   z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8:
    ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); nins.append(len(z))
  a=np.array(ics); print(h,'dates',len(a),'avgins',round(np.mean(nins),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4))
# turnover of cross sectional ranks at weekly-ish daily observations
ranks=f.rank(axis=1,pct=True); ch=(ranks.diff().abs().mean(axis=1)).dropna(); print('coverage',f.notna().mean().mean(),'turnover',ch.loc['2020-03-01':].mean())
# regimes
for a,b in [('2020-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-24')]:
  vals=[]
  for i in range(40,len(p)-20-1):
   if not(a<=str(p.index[i].date())<=b): continue
   z=pd.concat([f.iloc[i-1],np.log(p.iloc[i+20]/p.iloc[i])],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  x=np.array(vals); print('regime',a,b,'dates',len(x),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12),6))
# save signal artifacts for audit
out=f.loc[:end].dropna(how='all').reset_index().rename(columns={'index':'date'}); out.to_csv('scripts/miner_2_20311225_downside_adjusted_trend_signal.csv',index=False)
