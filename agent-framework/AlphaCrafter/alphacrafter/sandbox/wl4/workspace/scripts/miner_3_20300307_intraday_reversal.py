import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-03-06')
want=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in want:
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index();raw[s]=d.loc[:CUT]
px=pd.concat({s:d.close for s,d in raw.items()},axis=1).sort_index()
op=pd.concat({s:d.open for s,d in raw.items()},axis=1).reindex(px.index)
hi=pd.concat({s:d.high for s,d in raw.items()},axis=1).reindex(px.index);lo=pd.concat({s:d.low for s,d in raw.items()},axis=1).reindex(px.index)
intra=op.div(px.shift(1))-1
body=px.div(op)-1
atr=((hi-lo)/px.shift(1)).rolling(20,min_periods=15).median()
sig=(-(0.5*intra+body)/(atr+1e-9)).shift(1)
fwd={h:px.pct_change(h).shift(-h) for h in [1,5,10]}
def ev(y):
 aa=[];ns=[];turn=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
   if len(aa)>1:
    q=sig.loc[dt].rank(pct=True); q0=sig.shift(1).loc[dt].rank(pct=True);turn.append((q-q0).abs().mean())
 a=np.asarray(aa);return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.mean(turn)
for h,y in fwd.items():
 n,av,ic,ir,hit,t=ev(y);print(f'h={h} dates={n} avgN={av:.2f} IC={ic:.6f} ICIR={ir:.6f} hit={hit:.4f} turnover={t:.6f}')
print('assets',px.shape[1],'range',px.index.min().date(),px.index.max().date())
