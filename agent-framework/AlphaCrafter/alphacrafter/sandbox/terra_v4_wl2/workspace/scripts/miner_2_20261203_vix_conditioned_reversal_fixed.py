import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-03')
P=pd.concat([pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cut]
R=P.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill(); z=(v-v.rolling(60,min_periods=40).mean())/v.rolling(60,min_periods=40).std()
for h in [2,5,10]:
 raw=-R.rolling(h,min_periods=h).sum(); sig=raw.copy(); mask=(z>0).fillna(False); sig.loc[mask,:]=raw.loc[mask,:]; sig.loc[~mask,:]=raw.loc[~mask,:]*-.25
 vals=[]; dates=[]; n=[]; regimes=[]
 fwd=R.shift(-1)
 for dt in sig.index:
  q=pd.concat([sig.loc[dt].rename('f'),fwd.loc[dt].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.r.nunique()>1:
   vals.append(spearmanr(q.f,q.r).statistic);dates.append(dt);n.append(len(q));regimes.append(bool(mask.loc[dt]))
 a=np.array(vals); print(h,len(a),round(np.mean(n),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4),round(sig.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
 for rg in [False,True]:
  x=a[np.array(regimes)==rg]; print(' regime',rg,len(x),round(x.mean(),6) if len(x) else None)
