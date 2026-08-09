import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for fn in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(fn)[:-4]
 if s in keep:
  q=pd.read_csv(fn); q.date=pd.to_datetime(q.date); d[s]=q.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2032-09-29']; r=px.pct_change()
# One idea: residualized volatility-normalized path efficiency.
# efficiency measures directional smoothness; divide by realized volatility to prefer efficient moves per unit risk,
# then remove same-day cross-sectional component explained by ordinary risk-adjusted momentum.
ret20=px.pct_change(20); vol20=r.rolling(20,min_periods=15).std(); eff=r.rolling(20,min_periods=15).sum()/(r.abs().rolling(20,min_periods=15).sum()+1e-10)
raw=eff/(vol20+1e-10)
trend=ret20/(vol20+1e-10)
def resid(rowx,rowy):
 z=pd.concat([rowx,rowy],axis=1).dropna()
 if len(z)<8 or z.iloc[:,1].std()==0:return pd.Series(index=rowx.index,dtype=float)
 x=z.iloc[:,1].values; y=z.iloc[:,0].values
 b=np.cov(x,y,ddof=0)[0,1]/(np.var(x)+1e-12); a=y.mean()-b*x.mean()
 out=pd.Series(index=rowx.index,dtype=float); out.loc[z.index]=y-(a+b*x); return out
f=pd.DataFrame({dt:resid(raw.loc[dt],trend.loc[dt]) for dt in raw.index}).T.shift(1)
print('candidate=residualized_volatility_normalized_path_efficiency_20; universe=15; through=2032-09-29')
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
for label,a,b in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-32','2031','2032-09-29')]:
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],(px.shift(-1)/px-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(vals); print('REG',label,len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/(x.std(ddof=1)+1e-12),6))
print('coverage',round(f.notna().mean().mean(),4),'turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4),'meanvalid',round(f.notna().sum(axis=1).replace(0,np.nan).mean(),2))
print('library correlation audit: NOT COMPUTED (candidate fails admission contract pending exact signal reconstruction)')
