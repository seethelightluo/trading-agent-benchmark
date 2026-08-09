"""One idea: CN10Y transmission beta acceleration (20 versus 60 sessions).
Higher values mean an asset's exposure to CN10Y changes has recently fallen."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-12-13')
def close(a):
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date)
 return pd.to_numeric(d.set_index('date').loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:close(a) for a in A});r=p.pct_change(); cn=r['CN10Y']
def bet(a,w,mp):
 return r[a].rolling(w,min_periods=mp).cov(cn)/cn.rolling(w,min_periods=mp).var().replace(0,np.nan)
# A positive score is decreasing (more resilient) CN10Y transmission, standardized
# only cross-sectionally by Spearman IC, so no distributional assumption is required.
f=pd.DataFrame({a:bet(a,60,30)-bet(a,20,12) for a in A})
def metric(h):
 fw=p.shift(-h)/p-1; obs=[];ns=[]
 for d in p.index:
  z=pd.concat([f.loc[d].rename('factor'),fw.loc[d].rename('forward')],axis=1).dropna()
  if len(z)>=8:
   v=z.factor.corr(z.forward,method='spearman')
   if np.isfinite(v):obs.append((d,v));ns.append(len(z))
 x=pd.Series(dict(obs));sd=x.std(); turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 regs={}
 for name,mask in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028),('latest120',np.arange(len(x))>=len(x)-120)]:
  q=x[mask]; regs[name]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if len(q)>1 else np.nan,'hit_ratio':(q>0).mean() if len(q) else np.nan}
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_instruments':np.mean(ns),'turnover_10d':np.mean(turns),'regimes':regs}
print('FACTOR cn10y_transmission_beta_acceleration_20_60obs')
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'valid_cells',int(f.count().sum()),'of',f.size,'coverage',f.count().sum()/f.size)
for h in (1,5,10,20): print('METRIC',json.dumps(metric(h),default=float))
# Mandatory correlation is only decision-relevant if headline gates pass. These nearest
# admitted dependence/macro signals provide an early collinearity screen.
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
vol=r.rolling(20,min_periods=15).std()
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
spx=pd.DataFrame({a:-r[a].rolling(20,min_periods=12).cov(r.SPX)/r.SPX.rolling(20,min_periods=12).var().replace(0,np.nan) for a in A})
aut=r.rolling(20,min_periods=15).corr(r.shift(1))
for name,x in {'downside_peer_correlation_40obs':down,'negative_spx_beta_20obs':spx,'return_autocorrelation_20obs':aut}.items():
 z=pd.concat([f.stack().rename('candidate'),x.stack().rename('library')],axis=1).dropna()
 print('SCREEN_CORR',name,'rho',z.candidate.corr(z.library,method='spearman'),'cells',len(z))
