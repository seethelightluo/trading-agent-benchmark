import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
files=glob.glob('../persistent/stock_data/*.csv')
fields={}
for fld in ['open','high','low','close']:
 fields[fld]=pd.DataFrame({Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date')[fld] for f in files}).sort_index().ffill()
px=fields['close'][A]; o,h,l,c=[fields[x][A] for x in ['open','high','low','close']]
# Close-location pressure: signed intraday location, smoothed and normalized by its recent variability.
clv=((c-l)-(h-c))/(h-l).replace(0,np.nan)
f=clv.rolling(20,min_periods=12).mean()/clv.rolling(60,min_periods=30).std()
fw={n:px.shift(-n)/px-1 for n in [1,5,10,20]}
for n,y in fw.items():
 vals=[]; ns=[]; dates=[]
 for d in f.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:
   vals.append(spearmanr(f.loc[d][ok],y.loc[d][ok]).statistic);ns.append(ok.sum());dates.append(d)
 z=pd.Series(vals,index=dates)
 print('H',n,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'latest120',round(z.tail(120).mean(),6),round(z.tail(120).mean()/z.tail(120).std(ddof=1),6))
print('coverage',round(f.notna().mean().mean(),4),'turnover10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
y=fw[10]
for lab,mask in [('2020-23',f.index<'2024'),('2024-27',(f.index>='2024')&(f.index<'2028')),('2028-30',(f.index>='2028')&(f.index<'2031')),('2031',f.index>='2031')]:
 z=[]
 for d in f.index[mask]:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d][ok],y.loc[d][ok]).statistic)
 print('REG',lab,'dates',len(z),'IC',round(np.mean(z),6) if z else None,'ICIR',round(np.mean(z)/np.std(z,ddof=1),6) if len(z)>1 else None)
