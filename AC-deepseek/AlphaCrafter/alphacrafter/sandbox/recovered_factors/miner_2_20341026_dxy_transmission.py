import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
E=pd.Timestamp('2034-10-25')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}
P=pd.DataFrame(P).sort_index().loc[:E]
D=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
r=P.pct_change(); dr=D.pct_change()
# candidate: inverse DXY shock transmission, adjusted for own trend and volatility
x=dr.rolling(60,min_periods=40).apply(lambda z:(z.iloc[-1]-z.mean())/(z.std()+1e-12),raw=False)
f=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
    y=r[a]
    beta=y.rolling(30,min_periods=15).cov(dr)/dr.rolling(30,min_periods=15).var()
    vol=y.rolling(20,min_periods=15).std()
    trend=P[a].pct_change(20)
    # higher score = favorable when DXY shock transmission is low, residualized against trend/risk
    f[a]=-(beta*x).rolling(3,min_periods=1).mean() - .10*trend - .05*vol

def eval(h):
    vals=[]; ns=[]; dates=[]
    for d in P.index:
        if d>E: continue
        z=f.loc[d]; q=pd.Series({a:(P[a].dropna().shift(-h)/P[a].dropna()-1).reindex(P.index).loc[d] for a in A})
        ok=z.notna()&q.notna()
        if ok.sum()>=8:
            vals.append(spearmanr(z[ok],q[ok]).statistic);ns.append(ok.sum());dates.append(d)
    v=np.array(vals); return len(v),np.mean(ns),np.mean(v),np.mean(v)/(np.std(v,ddof=1)+1e-12),np.mean(v>0),dates[-1] if dates else None
for h in [1,5,10,20]: print(h,eval(h))
# regimes for 10d
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2034-10-25')]:
    # recompute from all and filter using helper-ish
    vals=[]
    for d in P.index:
      if not (pd.Timestamp(lo)<=d<=pd.Timestamp(hi)): continue
      z=f.loc[d];q=pd.Series({a:(P[a].dropna().shift(-10)/P[a].dropna()-1).reindex(P.index).loc[d] for a in A});ok=z.notna()&q.notna()
      if ok.sum()>=8: vals.append(spearmanr(z[ok],q[ok]).statistic)
    v=np.array(vals); print('regime',lo,hi,len(v),np.mean(v) if len(v) else np.nan,np.mean(v)/(np.std(v,ddof=1)+1e-12) if len(v)>1 else np.nan)
print('coverage',f.notna().sum().sum()/f.size,'assets',len(A),'dates',len(P))
# turnover rank
ranks=f.rank(axis=1,pct=True); print('turnover',np.nanmean(np.abs(ranks.diff()).mean(axis=1)))
