import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-10')
px={}
for s in U:
    d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
    px[s]=d['close'].loc[:end]
cl=pd.DataFrame(px).dropna(how='all')
# range efficiency: signed net movement divided by total absolute movement, 10d
ret=cl.pct_change()
eff=(cl.pct_change(10))/(ret.abs().rolling(10).sum())
# 10 trading-day forward return
fwd=cl.shift(-10)/cl-1
ics=[]; dates=[]; turnovers=[]; cov=[]
prev=None
for dt in eff.index:
    x=eff.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        if np.isfinite(ic): ics.append(ic); dates.append(dt); cov.append(len(z)/15)
        ranks=x.rank(pct=True)
        if prev is not None: turnovers.append((ranks-prev).abs().mean())
        prev=ranks
arr=np.array(ics)
print('factor=range_efficiency_10d dates=%d instruments=15 coverage=%.3f' % (len(arr),np.mean(cov)))
print('IC=%.5f ICIR=%.5f hit=%.3f mean_abs=%.5f turnover=%.5f' % (arr.mean(),arr.mean()/arr.std(ddof=1),np.mean(arr>0),np.mean(np.abs(arr)),np.nanmean(turnovers)))
for a,b in [(0, int(len(arr)*.33)),(int(len(arr)*.33),int(len(arr)*.66)),(int(len(arr)*.66),len(arr))]:
 q=arr[a:b]; print('regime',a,b,'n',len(q),'IC %.5f ICIR %.5f hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0)))
for h in [1,5,10,20]:
 yy=cl.shift(-h)/cl-1; aa=[]
 for dt in eff.index:
  z=pd.concat([eff.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'n',len(aa),'IC',np.nanmean(aa))
