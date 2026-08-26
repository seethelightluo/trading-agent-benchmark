import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for a in assets:
    p=f'../persistent/stock_data/{a}.csv'
    if not os.path.exists(p): p=f'../persistent/index_data/{a}.csv'
    d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
    frames[a]=d['close'].replace(0,np.nan)
px=pd.concat(frames,axis=1).sort_index().ffill()
# one interpretable idea: medium-term trend divided by recent risk
ret20=px.pct_change(20); ret60=px.pct_change(60); vol20=px.pct_change().rolling(20).std()*np.sqrt(252)
signal=ret60/vol20
fwd=px.shift(-10)/px-1
ics=[]; counts=[]; turns=[]
for dt in signal.index:
    x=signal.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
        ics.append(spearmanr(x[ok],y[ok]).statistic); counts.append(ok.sum())
# turnover: rank ordering changes, averaged daily
r=signal.rank(axis=1,pct=True); turns=((r-r.shift(1)).abs().mean(axis=1)).dropna()
def report(mask,name):
    z=np.array([v for v,m in zip(ics,mask) if m])
    # masks keyed dates after valid construction
    if len(z): print(name,len(z),f'IC {z.mean():.6f} ICIR {z.mean()/z.std(ddof=1):.6f} hit {(z>0).mean():.4f}')
valid_dates=[d for d in signal.index if signal.loc[d].notna().sum()>=8 and fwd.loc[d].notna().sum()>=8]
print('period',signal.index.min().date(),signal.index.max().date(),'valid_dates',len(ics),'mean_n',np.mean(counts),'coverage',np.mean(counts)/15)
print('full IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(np.mean(ics),np.mean(ics)/np.std(ics,ddof=1),(np.array(ics)>0).mean(),turns.mean()))
for cutoff in ['2026-07-16','2027-01-01','2028-06-14','2029-01-01']:
 m=np.array(valid_dates)>=pd.Timestamp(cutoff); z=np.array(ics)[m]
 print(cutoff,'dates',len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
for h in [1,5,10,20,40]:
 fw=px.shift(-h)/px-1; zz=[]
 for dt in signal.index:
  ok=signal.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8: zz.append(spearmanr(signal.loc[dt][ok],fw.loc[dt][ok]).statistic)
 print('horizon',h,'dates',len(zz),'IC %.6f ICIR %.6f'%(np.mean(zz),np.mean(zz)/np.std(zz,ddof=1)))
