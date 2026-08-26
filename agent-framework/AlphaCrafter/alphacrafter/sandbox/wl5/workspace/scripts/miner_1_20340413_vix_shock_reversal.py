import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
rz=(vix-vix.rolling(60,min_periods=40).mean())/vix.rolling(60,min_periods=40).std()
shock=(rz>1).astype(float)
trend=r.rolling(20).sum(); vol=r.rolling(40).std()*np.sqrt(20)
# VIX shock-conditioned reversal: contrarian in normal regimes, trend-following after an extreme VIX shock.
f=trend.div(vol.replace(0,np.nan))*(2*shock.values[:,None]-1)
fr=p.shift(-10).div(p)-1
rows=[]
for d in f.index:
 ok=f.loc[d].notna()&fr.loc[d].notna()
 if ok.sum()>=8: rows.append((d,spearmanr(f.loc[d,ok],fr.loc[d,ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-04-01':'2034-04-03']
mean=z.ic.mean(); sd=z.ic.std(ddof=1); icir=mean/sd*np.sqrt(252)
turnover=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[z.index].mean()
print('dates',len(z),'meanN',z.n.mean(),'instruments',len(assets),'coverage',f.loc[z.index].notna().stack().mean(),'IC10',mean,'ICIR',icir,'hit',(z.ic>0).mean(),'turnover',turnover)
for h in [5,10,20]:
 yy=p.shift(-h).div(p)-1; vals=[]
 for d in f.index:
  ok=f.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[d,ok],yy.loc[d,ok]).statistic)
 print('decay',h,np.nanmean(vals),len(vals))
for s,e in [('2025','2027-12-31'),('2028','2029-12-31'),('2030','2032-12-31'),('2033','2034-04-03')]: print('regime',s,e,z.loc[s:e].ic.mean(),len(z.loc[s:e]))
out=f.loc[z.index].stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20340413_vix_shock_reversal_signal.csv',index=False); print('artifact rows',len(out))
