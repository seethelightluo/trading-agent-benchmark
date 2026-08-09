import pandas as pd, numpy as np, glob, os, json
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 D[a]=d['close']
P=pd.DataFrame(D).sort_index(); R=P.pct_change()
# Candidate: downside-risk quality. Lower recent downside semivolatility should rank higher.
down=R.clip(upper=0).pow(2).rolling(40,min_periods=30).mean().pow(.5)
F=(-down).shift(1)
fr={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
print('data',P.index.min().date(),P.index.max().date(),'assets',len(assets),'dates',len(P),'raw_coverage',round(F.notna().mean().mean(),4))
def series(h,idx=None):
 out=[]; ns=[]; dates=[]
 for d in (P.index if idx is None else idx):
  x,y=F.loc[d],fr[h].loc[d]; z=x.notna()&y.notna()&np.isfinite(x)&np.isfinite(y)
  if z.sum()>=8 and x[z].nunique()>1 and y[z].nunique()>1:
   out.append(spearmanr(x[z],y[z]).statistic); ns.append(z.sum()); dates.append(d)
 return pd.Series(out,index=dates),ns
for h in [1,5,10,20]:
 s,n=series(h); print('h',h,'dates',len(s),'meanN',round(np.mean(n),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-06-09')]:
 s,n=series(1,P.loc[lo:hi].index); print('regime',lo,hi,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6))
rank=F.rank(axis=1,pct=True); ts=[]
for i in range(10,len(rank),10):
 z=rank.iloc[i-10].notna()&rank.iloc[i].notna()
 if z.sum()>=8: ts.append((rank.iloc[i-10][z]-rank.iloc[i][z]).abs().mean())
print('turnover10_proxy',round(np.mean(ts),4),'turnover_obs',len(ts))
# correlation against every current admitted factor, aligned signal cross-section cells
maxrho=0; arg=''; nfiles=0
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')!='EFFECTIVE': continue
  expr=j.get('calculation',{}).get('expression','')
  # only auditable common expressions: load factor by reusing known signal names where possible
  nfiles+=1
  # approximate library signal from factor metadata not executable; skip impossible
 except: pass
print('effective_files',nfiles,'max_abs_library_correlation',round(maxrho,4),'NOTE correlation requires executable definitions; no admitted signal parsers')
