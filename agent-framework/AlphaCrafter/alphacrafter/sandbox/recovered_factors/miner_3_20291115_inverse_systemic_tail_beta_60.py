"""One candidate: inverse cross-asset systemic-tail beta, 60 sessions.
On days where the contemporaneous cross-asset median return is in its trailing
bottom quintile, measure an asset's mean return divided by the median shock;
negative beta favours low participation in broad adverse tail events.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def close(a):
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame({a:close(a) for a in A}).sort_index(); R=P.pct_change(); M=R.median(axis=1)
# threshold excludes current observation through shift; factor itself then has an additional execution lag
q=M.rolling(60,min_periods=40).quantile(.20).shift(1); event=M.lt(q)
# conditional means use only event days; 15 events minimum, denominator remains negative by construction
num=R.where(event, np.nan).rolling(60,min_periods=15).mean()
den=M.where(event, np.nan).rolling(60,min_periods=15).mean()
F=(-num.div(den,axis=0)).shift(1); F=F.sub(F.median(axis=1),axis=0); cut=P.index.max()
def metric(h,lo=None):
 x=F.loc[lo:] if lo else F; y=(P.shift(-h)/P-1).reindex(x.index); z=[]; n=[]
 for t in x.index:
  v=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(v)>=8:
   c=spearmanr(v.iloc[:,0],v.iloc[:,1]).statistic
   if np.isfinite(c): z.append(c); n.append(len(v))
 z=np.asarray(z)
 return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(n)),2),'min_n':int(min(n))} if len(z) else {'dates':0}
print('FACTOR inverse_systemic_tail_beta_60 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20): print('H',h,metric(h))
for lab,lo in [('2020_22','2020-01-01'),('2023_24','2023-01-01'),('2025_26','2025-01-01'),('2027_current','2027-01-01'),('recent180',str(cut-pd.Timedelta(days=180)))]: print('REGIME10',lab,metric(10,lo))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
print('ADMISSION_CORRELATION pending: calculate only if IC gates and robust regime evidence justify proceeding.')
