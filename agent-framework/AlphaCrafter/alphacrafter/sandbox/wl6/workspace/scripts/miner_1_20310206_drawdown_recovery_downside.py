import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cutoff=pd.Timestamp('2031-02-05')
px={}
for a in assets:
 d=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); px[a]=d[d.index<=cutoff]
close=pd.DataFrame(px).sort_index(); ret=close.pct_change(); low=close.rolling(60,min_periods=50).min(); recovery=close/low-1
# downside semideviation remains defined for assets with few negative observations
neg2=ret.clip(upper=0).pow(2); down=np.sqrt(neg2.rolling(40,min_periods=25).mean()).replace(0,np.nan); factor=recovery/down
for h in [5,10,20]:
 fwd=close.shift(-h)/close-1; rows=[]; turnovers=[]; prev=None
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
   ranks=factor.loc[dt].rank(pct=True)
   if prev is not None: turnovers.append(np.nanmean(abs(ranks-prev)))
   prev=ranks
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 if len(q)==0: print('H',h,'NO OBS'); continue
 print('H',h,'dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std()*np.sqrt(252),'hit',(q.ic>0).mean(),'turnover',np.nanmean(turnovers))
 print('regimes',[(yr,round(q[q.index.year==yr].ic.mean(),4),len(q[q.index.year==yr])) for yr in range(2020,2032) if len(q[q.index.year==yr])])
