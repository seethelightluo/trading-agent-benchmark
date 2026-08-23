import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-01-07'); px={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.split('/')[-1][:-4]; d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index();px[s]=d[d.index<=cut]
P=pd.DataFrame(px).sort_index(); r=P.pct_change();
# defensive low-volatility signal: negative trailing 20d realized vol, lagged one day
fac=-r.shift(1).rolling(20).std(); fwd=P.pct_change().shift(-1)
for h in [1,2,5,10]:
 y=P.shift(-h)/P-1; ic=[]; ns=[]; cov=[]; prev=None; tos=[]
 for dt in P.index:
  x=fac.loc[dt]; yy=y.loc[dt]; ok=x.notna()&yy.notna()
  if ok.sum()>=8:
   ic.append(spearmanr(x[ok],yy[ok]).statistic);ns.append(ok.sum());cov.append(ok.mean())
   rr=x.rank(pct=True);tos.append(np.nan if prev is None else np.mean(abs(rr-prev)));prev=rr
 z=pd.Series(ic);print('h',h,'dates',len(z),'avg_n',np.mean(ns),'coverage',np.mean(cov),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turn',np.nanmean(tos))
