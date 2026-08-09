import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2034-09-13')
px={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); px[a]=d.loc[d.index<=E,'close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(fill_method=None)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float).reindex(p.index).ffill()
# Conditional VIX-residual rebound: recovery is favored only when VIX has recently fallen,
# and recent asset returns are residualized against VIX changes and scaled by idiosyncratic risk.
x=vix.pct_change(); beta=r.rolling(60,min_periods=40).cov(x).div(x.rolling(60,min_periods=40).var(),axis=0)
resid=r.rolling(20,min_periods=15).sum()-beta.mul(x.rolling(20,min_periods=15).sum(),axis=0)
idvol=(r-beta.mul(x,axis=0)).rolling(20,min_periods=15).std()
vix_regime=-(x.rolling(5,min_periods=4).sum())
# positive multiplier in calm/falling-VIX regimes, zero otherwise; fixed bounded transform
mult=vix_regime.clip(lower=0).div(vix_regime.abs().rolling(60,min_periods=40).median()+1e-8).clip(upper=3)
f=-resid/(idvol+1e-8)*mult
print('candidate conditional VIX-residual recovery; cutoff',E.date(),'rows',len(p),'assets',len(A))
for h in [1,5,10,20]:
 vals=[]; ns=[]; fr=p.shift(-h)/p-1
 for t in f.index:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 q=np.asarray(vals); print('H',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1),np.mean(q>0)))
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
fr=p.shift(-10)/p-1; vals=[]; ds=[]
for t in f.index:
 z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
 if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(t)
v=np.asarray(vals); ds=pd.Series(ds)
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-09-13'),('2033','2034-09-13')]:
 q=(ds>=lo)&(ds<=hi); y=v[q]; print('regime',lo,hi,'dates',len(y),'IC %.6f ICIR %.6f'%(y.mean(),y.mean()/y.std(ddof=1)) if len(y)>1 else 'nan')
print('library_audit FAILED: candidate not admitted without exact common-cell correlations against all active library signals')
