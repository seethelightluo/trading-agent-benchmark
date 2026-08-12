import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
frames={}
for a in assets:
 p=os.path.join(base,a+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); d=d[d.date<='2029-04-04']; d=d.set_index('date').sort_index(); frames[a]=d
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date); vix=vix.set_index('date').sort_index()
# VIX close, joined observation; use rolling median on history only
v=vix[vix.index<='2029-04-04'].close.reindex(sorted(set().union(*[set(x.index) for x in frames.values()]))).ffill()
med=v.rolling(126,min_periods=63).median()
rows=[]
for a,d in frames.items():
 r=d.close.pct_change(); vol=r.rolling(20,min_periods=20).std()
 raw=-(r.rolling(3,min_periods=3).sum())/(vol*np.sqrt(3)+1e-12)
 # stress-conditioned reversal vs calm momentum, using prior-complete day's VIX state
 z=pd.DataFrame({'raw':raw,'v':v,'med':med}).reindex(d.index).shift(1)
 f=np.where(z.v>z.med,-z.raw,z.raw) # raw is reversal; stress reversal, calm momentum
 f=pd.Series(f,index=d.index)
 # f at date t uses through t-1 due shift; forward close return t to t+1
 fr=d.close.pct_change().shift(-1)
 for dt in d.index: rows.append((dt,a,f.loc[dt],fr.loc[dt]))
x=pd.DataFrame(rows,columns=['date','asset','factor','fwd']).dropna()
ics=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: ics.append((dt,spearmanr(g.factor,g.fwd).statistic,len(g)))
ic=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date')
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',len(x)/sum(len(d) for d in frames.values()))
print('daily mean %.6f ICIR %.6f hit %.3f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(ddof=1), (ic.ic>0).mean()))
for h in [3,5,10]:
 rr=[]
 for a,d in frames.items():
  r=d.close.pct_change(h).shift(-h)
  # rederive factor matching
  q=d.close.pct_change(); vv=q.rolling(20,min_periods=20).std(); raw=-(q.rolling(3,min_periods=3).sum())/(vv*np.sqrt(3)+1e-12)
  z=pd.DataFrame({'raw':raw,'v':v.reindex(d.index),'med':med.reindex(d.index)}).shift(1)
  ff=pd.Series(np.where(z.v>z.med,-z.raw,z.raw),index=d.index)
  rr += [(dt,a,ff.loc[dt],r.loc[dt]) for dt in d.index]
 q=pd.DataFrame(rr,columns=['date','a','f','r']).dropna(); vals=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8: vals.append(spearmanr(g.f,g.r).statistic)
 vals=np.array(vals); print('h',h,'IC %.6f ICIR %.6f dates %d'%(np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals,ddof=1),len(vals)))
print('recent',ic[ic.index>='2027-01-01'].ic.mean(), '2028+',ic[ic.index>='2028-01-01'].ic.mean())
# monthly stability
print(ic.resample('YS').ic.mean().to_string())
