import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Trend acceleration: recent 20d return minus trailing 60d average daily return, risk scaled.
D={}
for s in U:
    x=get_stock_daily_data(s, days=4000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index(); R=P.pct_change()
# avoid lookahead: signal at t, forward return t->t+10
f=(P.shift(-10)/P-1)
fac=(P/P.shift(20)-1 - (P/P.shift(60)-1)/3) / (R.rolling(20).std()*np.sqrt(20)+1e-9)
# cross-sectional spearman IC, dates with >=8
ics=[]; n=[]; turnovers=[]
prev=None
for dt in fac.index:
    a=fac.loc[dt]; b=f.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ics.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank())); n.append(len(z))
    ranks=a.rank(pct=True)
    if prev is not None: turnovers.append(np.mean(abs(ranks-prev)))
    prev=ranks
ic=pd.Series(ics).dropna(); mean=ic.mean(); sd=ic.std(ddof=1); icir=mean/sd*np.sqrt(252/10) if sd else np.nan
print({'factor':'trend_acceleration_10d','dates':len(ic),'avg_instruments':np.mean(n),'coverage':np.mean(n)/15,'IC':mean,'ICIR':icir,'hit':np.mean(ic>0),'turnover_proxy':np.mean(turnovers)})
for y,g in pd.DataFrame({'ic':ic.values},index=[dt for dt in fac.index if len(pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna())>=8]).groupby(lambda x:x.year): print(y, len(g), g.ic.mean())
for h in [5,10,20]:
 ff=P.shift(-h)/P-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('decay',h,np.nanmean(vals),len(vals))
