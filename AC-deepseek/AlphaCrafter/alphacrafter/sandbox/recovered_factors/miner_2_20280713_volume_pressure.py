import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}
I=sorted(set.intersection(*[set(x.index) for x in D.values()]))
c=pd.DataFrame({a:D[a].reindex(I).close for a in A}); v=pd.DataFrame({a:D[a].reindex(I).volume for a in A})
r=c.pct_change()
# Volume-confirmed pressure: signed return weighted by abnormal volume, averaged over 10 sessions; lagged.
volratio=v/v.rolling(20,min_periods=10).median()
f=(r*volratio.clip(upper=5)).rolling(10,min_periods=10).sum().shift(1)
f=f.sub(f.median(axis=1),axis=0)
print('candidate volume_confirmed_pressure dates',len(I),'assets',len(A),'coverage',round(float(f.notna().mean().mean()),4))
allvals={}
for h in [1,5,10,20]:
 fr=c.pct_change(h).shift(-h); vals=[];ns=[];dates=[]
 for dt in I:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z));dates.append(dt)
 x=np.array(vals); allvals[h]=(x,dates)
 print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
 if h==1:
  for y in range(2020,2029):
   z=x[[d.year==y for d in dates]]
   if len(z): print('regime',y,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean().mean()),5))
# evidence using exact candidate-like signals from admitted factor names where reconstructable
lib={}
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p)); fid=j.get('factor_id',''); st=j.get('validation',{}).get('status')
  if st not in ('EFFECTIVE','ACTIVE',None): continue
  if 'risk_adjusted_trend' in fid or 'ravmom' in fid: x=(r.rolling(20).sum()/r.rolling(20).std()).shift(1)
  elif 'volscaled_trend' in fid: x=(r.rolling(10).sum()/r.rolling(20).std()-r.rolling(40).sum()/r.rolling(60).std()).shift(1)
  elif 'rank_acceleration' in fid: x=(r.rolling(5).sum().rank(axis=1,pct=True)-r.rolling(20).sum().rank(axis=1,pct=True)).shift(1)
  elif 'volatility_regime' in fid: x=(r.rolling(5).std()/r.rolling(60).std()).shift(1)
  elif 'volnorm_reversal' in fid: x=(-(r.rolling(5).sum()/r.rolling(5).std())).shift(1)
  else: continue
  lib[fid]=x
 except: pass
co=[]
for fid,x in lib.items():
 z=pd.concat([f.stack(),x.stack()],axis=1).dropna()
 if len(z)>100: co.append((fid,abs(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)))
print('library evidence count',len(co),'max_abs_library_correlation',round(max([x[1] for x in co],default=0),6),'top',sorted(co,key=lambda z:-z[1])[:5])
