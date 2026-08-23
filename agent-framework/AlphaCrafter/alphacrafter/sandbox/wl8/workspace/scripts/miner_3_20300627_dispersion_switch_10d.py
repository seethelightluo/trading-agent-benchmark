import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-06-27'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'];px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change();
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
base=-p.pct_change(5); mult=(disp/disp.rolling(120,min_periods=60).median()).clip(.5,2.0)
f=base.mul(mult,axis=0)
ics=[];ns=[];cs=[];tr=[];ds=[]
for i in range(len(p)-10):
 if p.index[i]<p.index[70] or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 v=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(v):
  ics.append(v);ns.append(ok.sum());cs.append(ok.mean());ds.append(p.index[i])
  if i:
   q=f.iloc[i-1];oo=x.notna()&q.notna();tr.append((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean())
a=np.array(ics);D=np.array(ds);print({'factor':'dispersion_conditioned_reversal_5d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cs)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(tr))})
for name,m in [('180',D>=pd.Timestamp('2029-12-01')),('360',D>=pd.Timestamp('2029-01-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[m];print(name,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':ds,'ic':a}).to_csv('scripts/miner_3_20300627_dispersion_switch_10d_ic.csv',index=False)
f.loc[pd.Index(ds)].to_csv('scripts/miner_3_20300627_dispersion_switch_10d_signal.csv')
