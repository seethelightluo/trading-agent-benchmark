"""One candidate: residual downside-event spacing conditional on next-session close-location recovery."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
START=pd.Timestamp('2026-07-16'); CUT=pd.Timestamp('2032-05-26')
def get(a, field):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[START:CUT,field].astype(float)
c=pd.DataFrame({a:get(a,'close') for a in A}); h=pd.DataFrame({a:get(a,'high') for a in A}); l=pd.DataFrame({a:get(a,'low') for a in A})
r=c.pct_change(fill_method=None); m=r.median(axis=1)
b=r.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0); res=r-b.mul(m,axis=0)
# A material idiosyncratic loss is followed (one completed session later) by its
# close-location recovery.  The signal is the 20d-versus-60d increase in the
# average age/spacing of such well-absorbed shocks (large = shocks becoming rarer).
rsd=res.rolling(60,min_periods=45).std()
event=res.lt(-rsd)
clv=(c-l).div((h-l).replace(0,np.nan)).clip(0,1)
qualified=event & clv.shift(-1).gt(.60) # used only after shifting below to completed time
# At date t, qualification for shock t-1 uses close-location t, hence no future use.
q=qualified.shift(1)
# days since most recent qualified shock, capped so an old missing event cannot dominate
age=pd.DataFrame(index=c.index,columns=A,dtype=float)
for a in A:
 last=np.nan; out=[]
 for i,x in enumerate(q[a].fillna(False).values):
  if x: last=i
  out.append(np.nan if np.isnan(last) else min(i-last,60))
 age[a]=out
recent=age.where(q).rolling(20,min_periods=4).mean()
base=age.where(q).rolling(60,min_periods=10).mean()
s=recent.div(base).sub(1)
print('FACTOR residual_downside_event_spacing_close_location_relief_20_60obs cutoff',CUT.date(),'assets',len(A),'calendar_dates',len(c))
print('formula: mean_20[age since qualified residual<-sd60 shock] / mean_60[same] - 1; qualification shock t-1 with close_location(t)>0.60')
print('coverage',int(s.notna().sum().sum()),'/',s.size,round(s.notna().mean().mean(),6))
O={}
for qh in (1,5,10,20):
 y=c.shift(-qh).div(c).sub(1); vals=[]; dates=[]; ns=[]
 for t in s.index:
  z=pd.concat([s.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(t);ns.append(len(z))
 vals=np.asarray(vals); dates=pd.DatetimeIndex(dates); O[qh]=(vals,dates)
 print('H',qh,'dates',len(vals),'IC',round(vals.mean(),6),'ICIR',round(vals.mean()/vals.std(ddof=1),6),'hit',round((vals>0).mean(),6),'mean_n',round(np.mean(ns),3),'min_n',min(ns),'PASS',abs(vals.mean())>=.007 and abs(vals.mean()/vals.std(ddof=1))>=.084)
for name,lo,hi in [('2026_2029','2026-07-16','2029-12-31'),('2030_cutoff','2030-01-01',CUT),('recent_12m','2031-05-27',CUT)]:
 x,d=O[20]; z=x[(d>=lo)&(d<=hi)]; print('REGIME',name,'H20 dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6))
rk=s.rank(axis=1,pct=True); print('TURNOVER',round((rk-rk.shift()).abs().stack().mean(),6),'median_iqr',round((s.quantile(.75,axis=1)-s.quantile(.25,axis=1)).median(),6))
print('NOVELTY pending: only perform exact full-library audit if predictive gate passes.')
