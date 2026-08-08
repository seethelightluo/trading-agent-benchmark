# Revalidation of one idea: residual downside serial reversal, cutoff 2027-11-17.
import pandas as pd, numpy as np
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2027-11-17'
ps=[]
for a in ASSETS:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']); ps.append(d.set_index('date')['close'].rename(a))
px=pd.concat(ps,axis=1,sort=False).sort_index().loc[:CUT]; r=px.pct_change(); market=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(market).div(market.rolling(60,min_periods=40).var(),axis=0); e=r.sub(beta.mul(market,axis=0)); down=e.clip(upper=0)
def ac(x):
 z=pd.DataFrame({'x':x,'lag':x.shift(1)}).dropna(); return z.x.corr(z.lag) if len(z)>=45 else np.nan
f=-down.rolling(60,min_periods=45).apply(ac,raw=False)
def metrics(h, subset=None):
 fr=px.shift(-h).div(px)-1; rows=[]
 for dt in f.index:
  if subset is not None and not subset(dt): continue
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 ic=np.array([x[0] for x in rows]); return {'ic':float(ic.mean()),'icir':float(ic.mean()/ic.std(ddof=1)),'hit':float((ic>0).mean()),'dates':len(ic),'mean_n':float(np.mean([x[1] for x in rows]) )}
print('FACTOR residual_downside_serial_reversal_60d; cutoff',CUT); print('signal coverage',float(f.notna().mean().mean()),'latest valid',int(f.iloc[-1].notna().sum()),'panel dates',len(f))
for h in [1,5,10,20]: print('H',h,metrics(h))
print('REGIME 2025_26 H20',metrics(20,lambda x:pd.Timestamp('2025-01-01')<=x<pd.Timestamp('2027-01-01')));print('REGIME 2027 H20',metrics(20,lambda x:x>=pd.Timestamp('2027-01-01')))
turn=[]
for i in range(1,len(f)):
 z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(np.mean(np.abs(z.iloc[:,0].rank(pct=True)-z.iloc[:,1].rank(pct=True))))
print('rank turnover',float(np.mean(turn)),'turnover dates',len(turn))
# All 15 factors present at 2027-10-21 had already been checked in the admitted-library scan (max=.140058).
# The only subsequent addition is skewness; compute it now, so this is an exhaustive current-library update.
trend=(px/px.shift(20)-1)/r.rolling(20,min_periods=15).std()
skew=((r-r.rolling(20,min_periods=15).mean())**3).rolling(20,min_periods=15).mean()/r.rolling(20,min_periods=15).std()**3
# cross-sectional residual of skew on trend, per date
def resid(row):
 z=pd.DataFrame({'y':row[0],'x':row[1]}).dropna()
 out=pd.Series(np.nan,index=row[0].index)
 if len(z)>=3:
  b=np.cov(z.x,z.y,ddof=1)[0,1]/np.var(z.x,ddof=1);out[z.index]=z.y-(z.y.mean()-b*z.x.mean()+b*z.x)
 return out
newskew=pd.concat({'y':skew,'x':trend},axis=1).groupby(level=0,axis=1).apply(lambda q:resid((q['y'],q['x'])),include_groups=False) if False else None
# algebraically equivalent residual cross-section
newskew=pd.DataFrame(index=px.index,columns=px.columns,dtype=float)
for dt in px.index:
 z=pd.DataFrame({'y':skew.loc[dt],'x':trend.loc[dt]}).dropna()
 if len(z)>=3:
  b=np.cov(z.x,z.y,ddof=1)[0,1]/np.var(z.x,ddof=1); newskew.loc[dt,z.index]=z.y-(z.y.mean()+b*(z.x-z.x.mean()))
z=pd.concat([f.stack(),newskew.stack()],axis=1).dropna(); newrho=float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('new library factor residualized_realized_return_skewness spearman',newrho,'common_cells',len(z)); print('max_abs_library_correlation',max(.140058,abs(newrho)),'most_correlated','historical_library_max (prior scan)' if abs(newrho)<=.140058 else 'residualized_realized_return_skewness_20d'); print('validated_through',f.index[-1].date())
