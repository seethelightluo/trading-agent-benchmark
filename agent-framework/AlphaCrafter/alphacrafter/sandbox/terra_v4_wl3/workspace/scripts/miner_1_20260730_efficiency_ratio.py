import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
prices={}
for f in files:
 s=os.path.basename(f).replace('.csv',''); d=pd.read_csv(f); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date'); d=d[d['date']<='2026-07-15'].set_index('date'); prices[s]=d
# efficiency ratio: directional displacement / path length, signed by trend
rows=[]
for s,d in prices.items():
 r=d['close'].pct_change(); disp=d['close'].pct_change(20).abs(); path=r.abs().rolling(20).sum(); trend=np.sign(d['close'].pct_change(20)); fac=(disp/path*trend).replace([np.inf,-np.inf],np.nan); fwd=d['close'].pct_change().shift(-1)
 x=pd.DataFrame({'factor':fac,'fwd':fwd,'asset':s}); x['date']=x.index; rows.append(x.reset_index(drop=True))
z=pd.concat(rows).dropna(subset=['factor','fwd'])
ics=[]; turnovers=[]; counts=[]
for dt,g in z.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>2 and g.fwd.nunique()>2:
  ics.append(spearmanr(g.factor,g.fwd).statistic); counts.append(len(g))
# rank turnover among consecutive dates, average overlap
rank=z.pivot(index='date',columns='asset',values='factor').rank(axis=1,pct=True)
turn=(rank.diff().abs().mean(axis=1)).dropna().mean()
a=np.array(ics)
print('dates',len(a),'avg_names',np.mean(counts),'coverage',len(z)/sum(len(d) for d in prices.values()),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'turnover',turn)
for h in [5,10]:
 rr=[]
 for s,d in prices.items():
  f=d['close'].pct_change(20).abs()/d['close'].pct_change().abs().rolling(20).sum()*np.sign(d['close'].pct_change(20)); fw=d['close'].pct_change(h).shift(-h); q=pd.DataFrame({'f':f,'y':fw}).dropna(); q['date']=q.index; rr.append(q.reset_index(drop=True))
 q=pd.concat(rr); vals=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8: vals.append(spearmanr(g.f,g.y).statistic)
 vals=np.array(vals); print(h,'d IC',vals.mean(),'ICIR',vals.mean()/vals.std(ddof=1),'dates',len(vals))
# annual
for yr,g in z.assign(year=z.date.dt.year).groupby('year'):
 vals=[]
 for dt,x in g.groupby('date'):
  if len(x)>=8: vals.append(spearmanr(x.factor,x.fwd).statistic)
 print(yr,len(vals),round(np.nanmean(vals),4),round(np.nanmean(vals)/np.nanstd(vals,ddof=1),4))
