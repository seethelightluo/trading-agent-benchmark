import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-07-18'); root='../persistent/stock_data/'
D={}
for a in A:
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 D[a]=d.loc[d.index<=END]
C=pd.concat({a:pd.to_numeric(D[a]['close'],errors='coerce') for a in A},axis=1).sort_index()
V=pd.concat({a:pd.to_numeric(D[a]['volume'],errors='coerce') for a in A},axis=1).sort_index()
# one interpretable idea: reversal of 10d return, amplified when recent volume participation is high,
# normalized by 20d volatility and demeaned cross-sectionally
r=C.pct_change(); vol=r.rolling(20,min_periods=15).std(); ret10=C/C.shift(10)-1
vp=V/(V.rolling(40,min_periods=20).median())
vp=vp.clip(0.25,4.0)
raw=-(ret10/vol)*(0.5+0.5*vp.rolling(5,min_periods=3).mean())
S=raw.sub(raw.median(axis=1),axis=0)
# winsorize cross-section
lo=S.quantile(.1,axis=1); hi=S.quantile(.9,axis=1); S=S.clip(lo,hi,axis=0)
for h in [1,5,10,20]:
  fwd=C.shift(-h)/C-1; vals=[]; dates=[]; ns=[]
  for dt in S.index:
   x=S.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
   if ok.sum()>=8:
    vals.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt); ns.append(ok.sum())
  z=np.array(vals); ic=np.nanmean(z); sd=np.nanstd(z,ddof=1); icir=ic/sd*np.sqrt(252/h) if sd else np.nan
  print(f'H{h}: IC={ic:.6f} ICIR={icir:.6f} dates={len(z)} hit={np.mean(z>0):.4f} meanN={np.mean(ns):.2f}')
  for loY,hiY in [('2025-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2035-07-18')]:
   q=np.array([v for v,d in zip(vals,dates) if pd.Timestamp(loY)<=d<=pd.Timestamp(hiY)])
   print(' ',loY[:4]+'-'+hiY[:4],len(q),f'{np.mean(q):.6f}',f'{np.mean(q)/np.std(q,ddof=1)*np.sqrt(252/h) if len(q)>1 else np.nan:.6f}')
# turnover and coverage
ranks=S.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).dropna()
print('cells',int(S.notna().sum().sum()),'coverage',float(S.notna().sum().sum()/S.size),'turnover',turnover.mean(),'rows',len(C),'assets',len(A))
# decay summary
