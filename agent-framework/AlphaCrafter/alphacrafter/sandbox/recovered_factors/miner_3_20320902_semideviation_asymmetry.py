import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for f in files:
 s=os.path.basename(f)[:-4]
 if s in keep:
  x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); d[s]=x.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2032-09-01']
r=px.pct_change()
# One interpretable idea: upside/downside semideviation asymmetry, lagged one day
up=r.where(r>0,0).rolling(40,min_periods=25).std()
dn=(-r.where(r<0,0)).rolling(40,min_periods=25).std()
f=(up/(dn+1e-8)).shift(1)
# forward returns
outs=[]
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/(np.std(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4))
# regimes one-day
fr=px.shift(-1)/px-1
for label,a,b in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-32','2031','2032-09-01')]:
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(vals); print(label,len(x),round(x.mean(),6),round(x.mean()/(x.std(ddof=1)+1e-12),6))
print('coverage',f.notna().mean().mean(),'turn10',f.rank(axis=1,pct=True).diff(10).abs().mean().mean())
