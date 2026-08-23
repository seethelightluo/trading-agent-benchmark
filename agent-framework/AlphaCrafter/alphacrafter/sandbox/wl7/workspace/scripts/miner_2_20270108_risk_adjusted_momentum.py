import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-01-07')
files=glob.glob('../persistent/stock_data/*.csv')
px={}
for f in files:
 s=f.split('/')[-1][:-4]; d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=cut]
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# standardized risk-adjusted medium momentum, computed at t and predicting t+1
mom=P.shift(1).pct_change(20); vol=r.shift(1).rolling(20).std(); fac=mom/vol
fwd=P.pct_change().shift(-1)
ics=[]; turnovers=[]; cov=[]; nms=[]
prev=None
for dt in P.index:
 x=fac.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],y[ok]).statistic
  ics.append(ic); cov.append(ok.mean()); nms.append(ok.sum())
  rank=x.rank(pct=True); turnovers.append(np.nan if prev is None else np.mean(abs(rank-prev)))
  prev=rank
z=pd.Series(ics).dropna(); print('cutoff',cut.date(),'dates',len(z),'avg_n',np.mean(nms),'coverage',np.mean(cov),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.nanmean(turnovers))
for h in [2,5,10]:
 yy=P.shift(-h)/P-1; q=[]
 for dt in P.index:
  ok=fac.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:q.append(spearmanr(fac.loc[dt][ok],yy.loc[dt][ok]).statistic)
 q=pd.Series(q);print('h',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
print('early/late',z.iloc[:len(z)//2].mean(),z.iloc[len(z)//2:].mean())
