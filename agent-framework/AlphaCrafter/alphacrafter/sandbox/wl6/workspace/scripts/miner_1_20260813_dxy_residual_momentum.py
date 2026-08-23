import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; macro='../persistent/index_data/DXY.csv'
def load(p):
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); d=d.set_index('date'); c='close'; return d[c].astype(float).sort_index().pct_change()
r={s:load(os.path.join(base,s+'.csv')) for s in U}; dxy=load(macro)
R=pd.DataFrame(r).join(dxy.rename('DXY'),how='inner').sort_index(); R=R.loc[:'2026-07-15']
# residual medium-term momentum: 20d asset return less rolling 60d beta times DXY 20d return
out=[]
for t in range(80,len(R)-1):
 hist=R.iloc[t-60:t]; cur=R.iloc[t-20:t]
 vals={}
 for s in U:
  x=hist[s]; z=hist.DXY; ok=x.notna()&z.notna()
  if ok.sum()<45: continue
  beta=np.cov(x[ok],z[ok],ddof=1)[0,1]/(np.var(z[ok],ddof=1)+1e-12)
  ar=cur[s].dropna().sum(); dr=cur.DXY.dropna().sum()
  if np.isfinite(beta) and np.isfinite(ar) and np.isfinite(dr): vals[s]=ar-beta*dr
 f=pd.Series(vals); fr=R.iloc[t+1][U]
 q=pd.concat([f,fr.rename('y')],axis=1).dropna()
 if len(q)>=8:
  ic=spearmanr(q.iloc[:,0],q.y).statistic
  out.append((R.index[t],ic,len(q),f))
ics=np.array([x[1] for x in out]);
print('dates',len(out),'avg_names',np.mean([x[2] for x in out]),'IC',ics.mean(),'ICIR',ics.mean()/(ics.std(ddof=1)+1e-12),'hit',(ics>0).mean())
for h in [5,10]:
 # same factor, forward compounded returns from t+1 through t+h
 vals=[]
 for t in range(80,len(R)-h):
  hist=R.iloc[t-60:t]; cur=R.iloc[t-20:t]; fs={}
  for s in U:
   x=hist[s];z=hist.DXY;ok=x.notna()&z.notna()
   if ok.sum()>=45:
    b=np.cov(x[ok],z[ok],ddof=1)[0,1]/(np.var(z[ok],ddof=1)+1e-12)
    fs[s]=cur[s].dropna().sum()-b*cur.DXY.dropna().sum()
  y=R.iloc[t+1:t+h+1][U].sum(); q=pd.concat([pd.Series(fs),y.rename('y')],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.y).statistic)
 a=np.array(vals);print('horizon',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
# turnover rank changes
ranks=[]
for _,_,_,f in out: ranks.append(f.rank(pct=True))
turn=[]
for a,b in zip(ranks[:-1],ranks[1:]): turn.append((a-b).abs().mean())
print('turnover',np.nanmean(turn),'coverage',np.mean([n/15 for _,_,n,_ in out]))
for yr in sorted(set(x[0].year for x in out)):
 a=np.array([x[1] for x in out if x[0].year==yr]); print('regime',yr,len(a),a.mean(),a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
