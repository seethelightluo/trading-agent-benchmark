import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-02-20')
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.split('/')[-1][:-4]; d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); raw[s]=d.loc[:CUT]
# daily close-to-close return and normalized true range; only available cross-asset files
px=pd.concat({s:d['close'] for s,d in raw.items()},axis=1).sort_index(); op=pd.concat({s:d['open'] for s,d in raw.items()},axis=1).reindex(px.index); hi=pd.concat({s:d['high'] for s,d in raw.items()},axis=1).reindex(px.index); lo=pd.concat({s:d['low'] for s,d in raw.items()},axis=1).reindex(px.index)
r=px.pct_change(); tr=(hi-lo)/px.shift(1)
# Reversal after an unusually large signed close-to-close shock, scaled by recent risk.
shock=r.rolling(3,min_periods=3).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(3))
range_state=(tr.rolling(3,min_periods=3).mean()/tr.rolling(60,min_periods=40).median()).clip(.5,3)
sig=(-shock*range_state).shift(1); fwd={h:px.pct_change(h).shift(-h) for h in [1,5,10]}
def ev(y):
 a=[]; ns=[]; tv=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
   if dt in sig.index[1:]:
    q=sig.loc[dt].rank(pct=True); q0=sig.shift(1).loc[dt].rank(pct=True); tv.append((q-q0).abs().mean())
 a=np.asarray(a); return len(a),np.mean(ns),np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.mean(tv)
for h,y in fwd.items():
 n,av,ic,ir,hit,t=ev(y); print(f'h={h} dates={n} avgN={av:.2f} IC={ic:.6f} ICIR={ir:.6f} hit={hit:.4f} turnover={t:.6f}')
 if h==1:
  for k in [250,500]:
   # recompute observations for recent window
   aa=[]
   for dt in sig.index:
    z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
    if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
   q=np.asarray(aa)[-k:]; print(f'recent{k} IC={q.mean():.6f} ICIR={q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(len(q)):.6f}')
print('assets',px.shape[1],'date_range',px.index.min().date(),px.index.max().date())
