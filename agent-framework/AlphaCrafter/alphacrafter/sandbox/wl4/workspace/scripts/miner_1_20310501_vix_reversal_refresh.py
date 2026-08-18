import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-05-01')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for s in U}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.loc[:cut]
dates=sorted(set.intersection(*[set(x.index) for x in px.values()]) & set(vix.index)); P=pd.DataFrame({s:px[s].reindex(dates) for s in U}); V=vix.reindex(dates)
# Lagged VIX percentile gates reversal: continuous multiplier, no future data.
vr=V.rolling(60,min_periods=30).rank(pct=True).shift(1)
f=(-P.pct_change(5).mul(0.5+vr,axis=0)).shift(1)
def run(H,start=0):
 a=[];ns=[]
 for i in range(start,len(P)-H-1):
  x=f.iloc[i]; y=P.iloc[i+1+H]/P.iloc[i+1]-1; ok=x.notna()&y.notna(); ns.append(int(ok.sum()))
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
   z=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(z): a.append(z)
 a=np.asarray(a); return len(a),a.mean(),a.mean()/a.std(ddof=1), (a>0).mean(),np.mean(ns),f.notna().mean().mean()
print('cutoff',cut.date(),'dates',len(P),'assets',len(U),'coverage',V.notna().mean())
for H in [1,5,10,20]: print('H',H,run(H))
for n in [365,730,1095]: print('recent',n,'H10',run(10,max(0,len(P)-n-11)))
print('turnover_proxy',((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.10).mean()))
