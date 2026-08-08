import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for fn in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(fn)[:-4]
 if s in keep:
  q=pd.read_csv(fn); q.date=pd.to_datetime(q.date); d[s]=q.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2033-04-27']; r=px.pct_change()
# Trend persistence: medium-term return divided by realized volatility, penalized by peak drawdown.
# All rolling inputs are shifted one day before forward-return test.
mom=px.pct_change(60)
vol=r.rolling(40,min_periods=30).std()*np.sqrt(40)
dd=px/px.rolling(120,min_periods=80).max()-1
sig=((mom/vol)*(1+dd.clip(-1,0))).shift(1)
print('candidate trend_drawdown_adjusted dates',len(px),'assets',len(px.columns))
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(sig.notna().mean().mean(),4),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
# yearly/regime blocks for daily horizon
h=1;fr=px.shift(-h)/px-1;a=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
x=pd.DataFrame(a,columns=['date','ic']);x['year']=x.date.dt.year
print('yearly',x.groupby('year').ic.agg(['count','mean']).round(5).to_string())
