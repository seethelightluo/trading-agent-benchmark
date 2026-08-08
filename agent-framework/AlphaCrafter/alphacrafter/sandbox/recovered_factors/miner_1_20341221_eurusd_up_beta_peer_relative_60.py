import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# one idea: EURUSD-up beta resilience: peer-relative return sensitivity to EURUSD return over 60 sessions
D=[]
for a in AS:
 p=f'../persistent/stock_data/{a}.csv'
 x=pd.read_csv(p,usecols=['date','close']).rename(columns={'close':a}).set_index('date');D.append(x)
P=pd.concat(D,axis=1).sort_index(); P.index=pd.to_datetime(P.index)
M=pd.read_csv('../persistent/index_data/EURUSD.csv',usecols=['date','close']).set_index('date').close;M.index=pd.to_datetime(M.index)
r=P.pct_change(); mr=M.reindex(P.index).pct_change()
# signal is rolling covariance of asset peer-relative return with standardized EURUSD daily change
rel=r.sub(r.median(axis=1),axis=0); z=(mr-mr.rolling(60,min_periods=40).mean())/mr.rolling(60,min_periods=40).std()
sig=rel.mul(z,axis=0).rolling(60,min_periods=40).mean().sub(rel.mul(z,axis=0).rolling(60,min_periods=40).mean().median(axis=1),axis=0).shift(1)
def stats(h):
 fwd=P.shift(-h)/P-1; ics=[]; breadth=[]
 for dt in sig.index:
  x=sig.loc[dt];y=fwd.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8: ics.append(spearmanr(x[ok],y[ok]).statistic);breadth.append(ok.sum())
 a=np.array(ics);return len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean(breadth),np.min(breadth)
print('CANDIDATE eurusd_up_beta_peer_relative_60; cutoff',P.index.max().date(),'cells',sig.notna().sum().sum(),'/',sig.size,'coverage',sig.notna().mean().mean())
for h in [1,5,10,20]:print('H',h,stats(h))
# period sign test h10
for nm,mask in [('2023_26',(sig.index>='2023-01-01')&(sig.index<'2027-01-01')),('2027_now',sig.index>='2027-01-01'),('recent180',sig.index>=sig.index.max()-pd.Timedelta(days=180))]:
 ss=sig.loc[mask];ff=(P.shift(-10)/P-1).loc[mask];a=[]
 for dt in ss.index:
  ok=ss.loc[dt].notna()&ff.loc[dt].notna()
  if ok.sum()>=8:a.append(spearmanr(ss.loc[dt][ok],ff.loc[dt][ok]).statistic)
 a=np.array(a);print(nm,len(a),a.mean(),a.mean()/a.std(ddof=1))
# turnover
rank=sig.rank(axis=1,pct=True);print('turnover',rank.diff().abs().stack().mean(),'dispersion',sig.std(axis=1).mean())
# conservative library correlation evidence: compare candidate to exact readily reconstructed active analogous macro signals only -- admission cannot be established without every signal
print('LIBRARY_CORRELATION_EVIDENCE unavailable: historical factor signal matrices are not persisted; candidate cannot be admitted under binding contract.')
