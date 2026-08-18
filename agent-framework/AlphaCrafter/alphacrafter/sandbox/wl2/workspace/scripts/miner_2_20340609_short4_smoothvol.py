import numpy as np
import pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
H=60
frames={}
for s in U:
    try: d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
    except Exception: continue
    if len(d)<200: continue
    d=d[['date','close']].drop_duplicates('date').set_index('date').sort_index()
    frames[s]=d.close
px=pd.DataFrame(frames).sort_index()
# signal uses only completed close at t and is explicitly lagged one session
r1=px.pct_change(); r2=px.pct_change(4); v20=r1.rolling(20).std(); v60=r1.rolling(60).std()
# Alternative: short reversal scaled by risk, with smooth volatility attenuation (bounded)
factor=-(r2/v20) * (v60/(v20+v60))
factor=factor.shift(1)
fwd=px.shift(-H)/px-1
ics=[]; turns=[]; cover=[]
for dt in factor.index:
    x=factor.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
    n=int(ok.sum())
    if n>=8:
        ics.append((dt, spearmanr(x[ok],y[ok]).statistic, n))
    prev=factor.shift(1).loc[dt]; ok2=x.notna()&prev.notna()
    if ok2.sum()>=8: turns.append(float((x[ok2].rank(pct=True)-prev[ok2].rank(pct=True)).abs().mean()))
    cover.append(n/len(U))
ic=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date')
# daily paper IC and ICIR, using mean/std of daily cross-sectional IC
mean=ic.ic.mean(); std=ic.ic.std(ddof=1); icir=mean/std*np.sqrt(252) if std else np.nan
# report both conventional daily ICIR and annualized; gate uses daily paper ICIR convention from prior cycles likely mean/std
paper_icir=mean/std if std else np.nan
print('assets',len(frames),'dates',len(ic),'avg_n',ic.n.mean(),'coverage',np.mean(cover),'turnover',np.mean(turns))
print('IC',mean,'paper_daily_ICIR',paper_icir,'annualized',icir,'hit',np.mean(ic.ic>0))
for a,b in [('2026-07-16','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2034-06-08')]:
 z=ic.loc[a:b].ic
 print(a,b,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
out=pd.DataFrame(factor.stack(),columns=['signal']); out.index.names=['date','symbol']; out.to_csv('scripts/miner_2_20340609_short4_smoothvol_signal.csv')
ic.to_csv('scripts/miner_2_20340609_short4_smoothvol_ic.csv')
