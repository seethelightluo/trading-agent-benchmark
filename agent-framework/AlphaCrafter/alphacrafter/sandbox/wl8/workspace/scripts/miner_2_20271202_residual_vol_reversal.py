import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-12-01'); P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date)
 P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# residual short-horizon reversal: remove contemporaneous cross-asset median move, normalize by lagged 20d risk
med=r.median(axis=1); resid=r.sub(med,axis=0)
sig=-(resid.rolling(3,min_periods=3).sum()/r.rolling(20,min_periods=18).std()).shift(1)
fwd=px.shift(-1)/px-1
A=[];N=[];D=[]; prev=None; T=[]
for d in sig.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:A.append(spearmanr(g.s,g.f).statistic);N.append(len(g));D.append(d)
 q=sig.loc[d].rank(pct=True)
 if prev is not None:T.append((q-prev).abs().mean())
 prev=q
A=np.array(A); D=pd.to_datetime(D)
def st(m):
 a=A[m]; return len(a),round(float(a.mean()),6),round(float(a.mean()/a.std(ddof=1)),6),round(float((a>0).mean()),4)
print('dates',len(A),'rows',sum(N),'avg_names',round(np.mean(N),2),'coverage',round(len(A)/len(sig),4),'turnover',round(float(np.nanmean(T)),4))
print('overall',st(np.ones(len(A),bool)))
for q,m in [('2020-22',D<'2023-01-01'),('2023-25',(D>='2023-01-01')&(D<'2026-01-01')),('2026',(D>='2026-01-01')&(D<'2027-01-01')),('2027',D>='2027-01-01'),('recent180',D>=END-pd.Timedelta(days=180))]: print(q,st(m))
for h in [3,5,10]:
 yy=px.shift(-h)/px-1; z=[]
 for d in sig.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':yy.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:z.append(spearmanr(g.s,g.f).statistic)
 z=np.array(z); print('horizon',h,'n',len(z),'IC',round(float(z.mean()),6),'ICIR',round(float(z.mean()/z.std(ddof=1)),6))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20271202_residual_vol_reversal_signal.csv',index=False)
