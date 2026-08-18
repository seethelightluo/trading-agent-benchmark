import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-05-01'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in U: raw[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
px=pd.concat({s:d.close for s,d in raw.items()},axis=1).sort_index(); r=px.pct_change()
# Reversal of recent weakness only when the longer trend is not deeply impaired.
ret10=px.pct_change(10); ret60=px.pct_change(60); dv=r.where(r<0).rolling(20,min_periods=15).std()
sig=(-(ret10/(dv+1e-12))*((ret60>-0.15).astype(float))).shift(1)
def ev(y,start=None):
 a=[];ns=[];tr=[]
 for dt in sig.index:
  if start and dt<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));tr.append((sig.loc[dt].rank(pct=True)-sig.shift(1).loc[dt].rank(pct=True)).abs().mean())
 a=np.array(a);return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.mean(tr)
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h)
 print('FULL',h,ev(y));print('RECENT',h,ev(y,'2029-05-01'))
print('assets',px.shape[1],'range',px.index.min().date(),px.index.max().date())