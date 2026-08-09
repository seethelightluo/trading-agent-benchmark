import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]
 if s in keep:
  x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); d[s]=x.set_index('date').close
px=pd.DataFrame(d).sort_index(); r=px.pct_change()
# Inverse of path efficiency, lagged one completed day: favors noisy/reversal-prone paths
pe=r.rolling(20,min_periods=15).sum()/(r.abs().rolling(20,min_periods=15).sum()+1e-10)
f=(-pe).shift(1)
print('candidate inverse_path_efficiency_20; universe=15; end',px.index.max().date())
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
# regime at H1 and H10
for h in [1,10]:
 fr=px.shift(-h)/px-1
 for label,a,b in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031',str(px.index.max().date()))]:
  vals=[]
  for dt in f.loc[a:b].index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  x=np.array(vals); print('REG',h,label,len(x),round(x.mean(),6),round(x.mean()/(x.std(ddof=1)+1e-12),6))
print('coverage',round(f.notna().mean().mean(),4),'turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
# correlations against admitted factor signals where expressions can be evaluated are not assumed; report explicit evidence unavailable here
print('library_correlation_audit: requires signal reconstruction; candidate not persisted unless independently audited')
