import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-02')
q={}
for s in U:
 try:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); q[s]=d[d.date<=CUT].set_index('date').close.astype(float)
 except Exception as e: print('missing',s,e)
p=pd.concat(q,axis=1).sort_index(); r=p.pct_change(); ret=p.pct_change(20); eff=ret.abs()/(r.abs().rolling(20,min_periods=15).sum()+1e-12); f=(ret*eff).shift(1)
def run(h):
 rr=p.pct_change(h); out=[]; ns=[]
 for i in range(len(p)-h):
  v=f.iloc[i].notna()&rr.iloc[i+h].notna()
  if v.sum()>=8 and f.iloc[i][v].nunique()>1 and rr.iloc[i+h][v].nunique()>1:
   out.append(spearmanr(f.iloc[i][v],rr.iloc[i+h][v]).statistic); ns.append(v.sum())
 x=np.array(out); return len(x),np.mean(ns),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0)
print('assets',len(q),'dates',len(p),'range',p.index.min(),p.index.max())
for h in [1,2,5,10,20]: print('horizon',h,run(h))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 ix=(p.index.year>=a)&(p.index.year<=b); old=f.index; f2=f.loc[ix]; p2=p.loc[ix]; r2=p2.pct_change(); rr=p2.pct_change(1); z=[]
 for dt in f2.index:
  if dt not in rr.index: continue
  v=f2.loc[dt].notna()&rr.loc[dt].notna()
  if v.sum()>=8:z.append(spearmanr(f2.loc[dt][v],rr.loc[dt][v]).statistic)
 print('regime',a,b,len(z),np.mean(z) if z else np.nan, (np.mean(z)/np.std(z,ddof=1)) if len(z)>1 else np.nan)
f.to_csv('scripts/miner_1_20270302_range_efficiency_signal.csv')
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
