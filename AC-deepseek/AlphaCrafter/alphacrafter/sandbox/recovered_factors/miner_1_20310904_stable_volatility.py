import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index()
r=p.pct_change()
# Stable-volatility factor: inverse variability of trailing realized volatility.
# It rewards assets whose 20d risk has been stable relative to their own 60d risk.
rv20=r.rolling(20,min_periods=15).std(); rvvol=rv20.rolling(60,min_periods=40).std()
f=-rvvol/(rv20.rolling(60,min_periods=40).mean()+1e-12)
print('dates',len(p),'assets',len(A),'range',p.index.min().date(),p.index.max().date())
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; z=[]; ns=[]; ds=[]
 for d in p.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   z.append(spearmanr(f.loc[d,ok],fw.loc[d,ok]).statistic);ns.append(ok.sum());ds.append(d)
 z=np.asarray(z); ds=np.asarray(ds,dtype='datetime64[ns]')
 print('H',h,'n_dates',len(z),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.3f'%(np.nanmean(z),np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12)*np.sqrt(len(z)),np.mean(z>0)))
 for lab,lo,hi in [('2020-23','2020','2023'),('2024-27','2024','2027'),('2028-30','2028','2030'),('2031','2031','2031')]:
  q=z[(ds>=np.datetime64(lo+'-01-01'))&(ds<=np.datetime64(hi+'-12-31'))]
  print(' ',lab,'n',len(q),'IC %.6f'%(np.nanmean(q) if len(q) else np.nan))
print('coverage %.4f'%f.notna().mean().mean(),'turnover10 %.4f'%f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
# basic distinctness audit against raw common signals, not library admission evidence
for name,s in [('ret5',p.pct_change(5)),('ret20',p.pct_change(20)),('invvol20',-rv20),('rvvol',rvvol)]:
 q=[]
 for d in p.index:
  ok=f.loc[d].notna()&s.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d,ok],s.loc[d,ok]).statistic)
 print('corr',name,'mean %.4f maxabs %.4f'%(np.nanmean(q),np.nanmax(np.abs(q))))
