import pandas as pd,numpy as np,glob,os,json
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; d={}
for fn in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(fn)[:-4]
 if s in keep:
  q=pd.read_csv(fn); q.date=pd.to_datetime(q.date); d[s]=q.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2033-05-11']; r=px.pct_change()
# Explicit reversal of acceleration: -(20d return - prior 40d return)/60d volatility.
acc=(px.pct_change(20)-px.pct_change(60))/ (r.rolling(60,min_periods=45).std()*np.sqrt(60))
sig=(-acc).shift(1)
print('candidate inverse_acceleration dates',len(px),'assets',len(px.columns))
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]; dates=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 a=np.array(a); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for name,lo,hi in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-12-31')]:
  x=a[[str(z.date())>=lo and str(z.date())<=hi for z in dates]]
  if len(x): print(' ',name,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),len(x))
print('coverage',round(sig.notna().mean().mean(),4),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
# decay daily through 20
for h in range(1,21):
 fr=px.shift(-h)/px-1;a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('DECAY',h,round(np.nanmean(a),6))
