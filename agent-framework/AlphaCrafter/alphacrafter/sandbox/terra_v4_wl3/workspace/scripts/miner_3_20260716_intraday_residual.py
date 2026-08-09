import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
rows=[]
for s,x in D.items():
 clv=-(2*(x.close-x.low)/(x.high-x.low).replace(0,np.nan)-1); intra=-(x.close/x.open-1); r=x.close.shift(-1)/x.close-1
 z=pd.DataFrame({'clv':clv,'intra':intra,'r':r});
 for dt,g in z.groupby(z.index):
  g=g.dropna()
  if len(g)>=8:
   # cross-sectional residual, orthogonal to CLV
   X=np.c_[np.ones(len(g)),g.clv.values]; b=np.linalg.lstsq(X,g.intra.values,rcond=None)[0]; f=g.intra.values-X@b
   rows.append((dt,spearmanr(f,g.r).statistic,spearmanr(g.intra,g.r).statistic,len(g),np.corrcoef(f,g.clv)[0,1]))
a=pd.DataFrame(rows,columns=['date','icres','icraw','n','corr']).set_index('date')
print('dates',len(a),'meanN',a.n.mean(),'raw',a.icraw.mean(),a.icraw.mean()/a.icraw.std(ddof=1),'res',a.icres.mean(),a.icres.mean()/a.icres.std(ddof=1),'hit',(a.icres>0).mean(),'corr',a.corr.mean())
for y,g in a.groupby(a.index.year): print(y,len(g),round(g.icres.mean(),4),round(g.icres.mean()/g.icres.std(ddof=1),4))
