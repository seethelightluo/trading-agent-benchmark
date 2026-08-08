import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
P=pd.DataFrame(px).sort_index(); R=np.log(P).diff(); vr=np.log(vix).diff().reindex(P.index)
# VIX shock-weighted short-term reversal; signal known at t, predicts t+1
for window in [2,3,5]:
    sig=-(R.rolling(window).sum()) * (vr.clip(lower=0).rolling(3).mean()) / (R.rolling(20).std()*np.sqrt(252))
    fwd=R.shift(-1)
    ics=[]; dates=[]; ns=[]
    for d in P.index:
        x=sig.loc[d]; y=fwd.loc[d]; ok=x.notna()&y.notna()
        if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
            ics.append(spearmanr(x[ok],y[ok]).statistic); dates.append(d); ns.append(ok.sum())
    z=np.array(ics); print('window',window,'dates',len(z),'meanN',np.mean(ns),'coverage',sig.notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
    for lo,hi in [('2020','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
      q=z[(np.array(dates)>=pd.Timestamp(lo+'-01-01'))&(np.array(dates)<=pd.Timestamp(hi+'-12-31'))]; print(lo,round(q.mean(),4) if len(q) else None,len(q))
# simple continuous reversal benchmark
PY
python scripts/miner_3_20340817_vix_spike_reversal.py