import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Acceleration: recent 5d return relative to its expected one-quarter of 20d trend,
# cross-sectionally demeaned, lagged one day to avoid lookahead.
px={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); px[s]=x.set_index('date').close
p=pd.DataFrame(px).sort_index()
r5=p.pct_change(5); r20=p.pct_change(20); vol=p.pct_change().rolling(20).std()
raw=r5-r20/4
# scale by volatility, then lag signal one completed day
f=(raw/vol.replace(0,np.nan)).shift(1)
# forward non-overlapping 10 trading-day return
fr=p.shift(-10)/p-1
rows=[]; sig=[]
for dt in f.index:
    a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(z)))
        sig.append((dt,*[f.loc[dt,s] if s in f.columns else np.nan for s in U]))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
valid=ic.ic.dropna(); n=len(valid); mean=valid.mean(); sd=valid.std(ddof=1)
# annualized ICIR convention sqrt(252/10)
print('dates',n,'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC10',mean,'ICIR10',mean/sd*np.sqrt(252/10),'hit', (valid>0).mean())
for h in [1,5,20,40]:
    ff=p.shift(-h)/p-1; vv=[]
    for dt in f.index:
      z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
      if len(z)>=8: vv.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    print('decay',h,np.nanmean(vv),len(vv))
# periods
for name,sub in [('early',valid.iloc[:n//3]),('middle',valid.iloc[n//3:2*n//3]),('late',valid.iloc[2*n//3:])]: print(name,sub.mean(),len(sub))
# turnover rank changes
r=f.rank(axis=1,pct=True); turn=(r-r.shift(1)).abs().mean(axis=1).mean()
print('rank_turnover',turn)
# artifacts
out=pd.DataFrame(sig,columns=['date']+U).set_index('date'); out.to_csv('scripts/miner_2_20300909_acceleration_volnorm_signal.csv')
ic.to_csv('scripts/miner_2_20300909_acceleration_volnorm_ic.csv')
