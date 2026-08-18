import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-10-05')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=end]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# 10-day signed trend efficiency, available at t, predict t+1
f=(p/p.shift(10)-1)/(r.abs().rolling(10).sum())
f=f.replace([np.inf,-np.inf],np.nan)
fr=r.shift(-1)
ics=[]; turnovers=[]; cov=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  cov.append(len(z)/15)
# signal turnover based ranks
rank=f.rank(axis=1,pct=True); turnovers=rank.diff().abs().mean(axis=1).dropna()
a=np.array(ics); print('end',end.date(),'dates',len(a),'assets',15,'coverage_mean',np.mean(cov),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'turnover',turnovers.mean())
# regime halves and horizons
for h in [1,3,5,10]:
 yy=p.shift(-h)/p-1; aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 aa=np.array(aa); print('h',h,'n',len(aa),'IC',aa.mean(),'ICIR',aa.mean()/aa.std(ddof=1),'hit',np.mean(aa>0))
for label,ix in [('early',a[:len(a)//2]),('late',a[len(a)//2:])]: print(label,len(ix),ix.mean(),ix.mean()/ix.std(ddof=1),np.mean(ix>0))
print('latest',f.loc[:end].tail(1).to_dict('records')[0])
