import pandas as pd, numpy as np, glob, json, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'; x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); D[a]=x.set_index('date').sort_index().close
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# efficiency-weighted directional trend: signed 40d return times path efficiency, lagged by one day
net=p.pct_change(40); path=r.abs().rolling(40,min_periods=30).sum(); eff=(net.abs()/path).clip(0,1); sig=(net*eff).shift(1)
# avoid scale issues and require valid
print('range',p.index.min(),p.index.max(),'assets',len(assets))
rows=[]
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; dates=[]; nvalid=[]
 for d in p.index:
  z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(d); nvalid.append(len(z))
 s=pd.Series(vals,index=dates); rows.append((h,len(s),np.mean(nvalid),s.mean(),s.mean()/s.std(),(s>0).mean()))
print('RESULTS h dates meanN IC ICIR hit')
for x in rows: print('%d %d %.2f %+.6f %+.6f %.3f'%x)
# turnover and coverage
rank=sig.rank(axis=1,pct=True); turnover=(rank-rank.shift(10)).abs().mean(axis=1).mean(); cov=sig.notna().sum(axis=1).mean()/15
print('coverage %.4f turnover10 %.4f mean_valid %.2f'%(cov,turnover,sig.notna().sum(axis=1).mean()))
# annual daily regime
fwd=p.shift(-1)/p-1
for y in sorted(set(p.index.year)):
 vals=[]
 for d in p.index[p.index.year==y]:
  z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 if len(vals)>20: print('YEAR',y,len(vals),np.mean(vals),np.mean(vals)/np.std(vals))
# library pooled correlations
maxrho=0; winner=''
for f in glob.glob('factors/*.json'):
 try:
  j=json.load(open(f)); expr=j.get('factor_id','')
  # only calculate against known signals unavailable; use current candidate correlation to common factor definitions impossible
 except: pass
print('MAX_LIBRARY_CORR_UNAVAILABLE until efficacy pass; candidate cells',int(sig.notna().sum().sum()))
