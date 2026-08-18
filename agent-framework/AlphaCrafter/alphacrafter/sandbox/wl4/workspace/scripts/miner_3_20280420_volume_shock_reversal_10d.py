import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}; vol={}
for a in ASSETS:
 f=f'{base}/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); px[a]=d.close; vol[a]=d.volume
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vol).reindex(P.index)
r=P.pct_change()
volz=(V-V.rolling(40,min_periods=20).mean())/V.rolling(40,min_periods=20).std()
rv=r.rolling(20,min_periods=15).std(); rev=-r.rolling(10,min_periods=10).sum()
shock=volz.shift(1).clip(-2,2)
factor=(rev.shift(1)/(rv.shift(1)*np.sqrt(20))).mul((1+0.35*shock).clip(0.3,1.7))
ics_by={h:[] for h in [1,5,10,20]}; valid_counts=[]
for dt in factor.index:
 for h in ics_by:
  y=P.shift(-h)/P-1; x=factor.loc[dt]; yy=y.loc[dt]; ok=x.notna()&yy.notna()
  if ok.sum()>=8:
   q=spearmanr(x[ok],yy[ok]).statistic
   if np.isfinite(q): ics_by[h].append(q)
for h in ics_by:
 s=pd.Series(ics_by[h]); print('horizon',h,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
s=pd.Series(ics_by[10]); print('10d thirds',[(len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6)) for z in np.array_split(s,3)])
y10=P.shift(-10); counts=[((factor.loc[d].notna())&y10.loc[d].notna()).sum() for d in factor.index]
print('assets',len(P.columns),'rows',len(P),'avg valid',round(float(np.mean(counts)),2))
