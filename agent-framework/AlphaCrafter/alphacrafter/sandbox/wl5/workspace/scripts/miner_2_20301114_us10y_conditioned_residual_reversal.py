import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,4000); return d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close.astype(float).sort_index()
P=pd.DataFrame({s:load(s) for s in U}).sort_index().loc[:pd.Timestamp('2030-11-13')]; R=P.pct_change(); m=R.mean(axis=1)
rm=P/P.shift(20)-1; mr=(1+m).rolling(20).apply(np.prod,raw=True); B=pd.DataFrame({s:R[s].rolling(60,min_periods=40).cov(m)/m.rolling(60,min_periods=40).var() for s in U}); res=rm-B.mul(mr,axis=0); vr=R-B.mul(m,axis=0); vol=vr.rolling(60,min_periods=40).std()*np.sqrt(252)
# observation-only US 10Y yield percentile as conditioning variable
macro=P['US10Y']; stress=macro.rolling(252,min_periods=100).rank(pct=True)
sig=(-res/vol).mul((0.5+stress),axis=0).replace([np.inf,-np.inf],np.nan); fwd=P.shift(-10)/P-1
rows=[];prev=None;turn=[];art=[]
for dt in sig.index:
 x=sig.loc[dt].dropna();y=fwd.loc[dt].reindex(x.index).dropna();x=x.reindex(y.index)
 if len(x)<8:continue
 rows.append((dt,x.corr(y,method='spearman'),len(x)));q=x.rank(pct=True)
 if prev is not None:turn.append((q-prev.reindex(q.index)).abs().mean())
 prev=q;art += [{'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)} for s,v in x.items()]
z=pd.DataFrame(rows,columns=['date','ic','n']).dropna();a=z.ic.mean();sd=z.ic.std(ddof=1);print('candidate us10y_conditioned_residual_reversal_20d');print('assets',15,'dates',len(z),'meanN',z.n.mean(),'coverage',z.n.mean()/15,'IC',a,'daily_ICIR',a/sd,'hit',(z.ic>0).mean(),'turnover',np.mean(turn))
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-11-13')]:
 q=z[(z.date>=x)&(z.date<=y)];print(x,y,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
pd.DataFrame(art).to_csv('scripts/miner_2_20301114_us10y_conditioned_residual_reversal_signal.csv',index=False)
