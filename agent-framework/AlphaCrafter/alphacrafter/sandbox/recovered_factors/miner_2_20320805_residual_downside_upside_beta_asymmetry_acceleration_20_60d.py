"""One-factor research: residual downside/upside market-beta asymmetry acceleration (20d vs 60d)."""
import numpy as np,pandas as pd,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-08-04'); cal=pd.bdate_range('2020-01-01',END)
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float)
p=pd.DataFrame({a:load(a).reindex(cal).ffill() for a in A}); r=p.pct_change(); m=r.mean(axis=1)
def b(x,d,w,n,mask):
 dd=d.where(mask); den=dd.rolling(w,min_periods=n).var()+1e-12
 return pd.DataFrame({a:x[a].where(mask).rolling(w,min_periods=n).cov(dd)/den for a in A})
# First remove each asset's broad-market beta.  The candidate measures whether its
# loss-day versus gain-day residual sensitivity has recently widened versus history.
base=pd.DataFrame({a:r[a].rolling(60,min_periods=42).cov(m)/(m.rolling(60,min_periods=42).var()+1e-12) for a in A})
e=r-base.mul(m,axis=0); down=m<0; up=m>0
def asym(w,n): return b(e,m,w,n,down)-b(e,m,w,n,up)
f=asym(20,8)-asym(60,25)
print('FACTOR residual_downside_upside_beta_asymmetry_acceleration_20_60d')
print('VALIDATION_END',END.date(),'PANEL',cal.min().date(),cal.max().date(),'UNIVERSE',len(A))
ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1); out=[]; ns=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z):out.append((t,z));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x; sd=x.std(ddof=1)
 print('HORIZON',h,json.dumps({'daily_paper_ic':round(float(x.mean()),6),'daily_paper_icir':round(float(x.mean()/sd),6),'ic_standard_error':round(float(sd/np.sqrt(len(x))),6),'ic_dates':len(x),'hit_ratio':round(float((x>0).mean()),6),'mean_valid_instruments':round(float(np.mean(ns)),3)}))
for name,mask in [('2020_2024',ics[10].index<pd.Timestamp('2025-01-01')),('2025_2026',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
 x=ics[10][mask];print('REGIME_10D',name,'dates',len(x),'ic',round(float(x.mean()),6),'icir',round(float(x.mean()/x.std(ddof=1)),6),'hit',round(float((x>0).mean()),6))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:ts.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(ts)),6),'TURNOVER_DATES',len(ts))
print('DECAY',json.dumps({str(k):round(float(v.mean()),6) for k,v in ics.items()}))
print('ORTHOGONALITY_STATUS PENDING: factor will not be persisted unless all admitted-factor signal correlations are reconstructed and max abs rho < 0.5.')
