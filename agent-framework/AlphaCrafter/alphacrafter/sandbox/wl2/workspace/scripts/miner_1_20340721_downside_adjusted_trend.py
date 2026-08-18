import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
p=pd.DataFrame(D).sort_index().ffill(); p=p.loc[:'2034-07-20']; r=p.pct_change()
# downside-adjusted trend: 30d return scaled by downside deviation, lagged
f=(p.pct_change(30)/r.where(r<0).rolling(30).std()).shift(1)
for h in [5,10,20]:
 a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],p.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(h,len(a),round(np.nanmean(a),6),round(np.nanmean(a)/np.nanstd(a,ddof=1),6),round(np.mean(a>0),4))
rank=f.rank(axis=1,pct=True);print('dates',len(p),'assets',len(U),'coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean(axis=1).mean())
for name,ix in [('early',f.index<'2025-01-01'),('mid',(f.index>='2025-01-01')&(f.index<'2030-01-01')),('recent',f.index>='2030-01-01')]:
 a=[]
 for dt in f.index[ix]:
  z=pd.concat([f.loc[dt],p.pct_change(10).shift(-10).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(name,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1))
print('period',p.index.min().date(),p.index.max().date())
