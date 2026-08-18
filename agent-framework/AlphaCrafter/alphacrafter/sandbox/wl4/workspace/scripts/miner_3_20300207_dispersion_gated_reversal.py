import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-02-06')
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.split('/')[-1][:-4]; d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); raw[s]=d.loc[:CUT]
px=pd.concat({s:d['close'] for s,d in raw.items()},axis=1).sort_index(); r=px.pct_change()
disp=r.std(axis=1)
state=(disp/disp.rolling(60,min_periods=40).median()).clip(.5,3.0)
base=-(r.rolling(5,min_periods=5).sum())/(r.rolling(20,min_periods=20).std()*np.sqrt(20))
sig=base.mul(state,axis=0).shift(1); fwd=r.shift(-1)
def evaly(yy):
 ics=[]; ns=[]; turns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
   q=sig.loc[dt].rank(pct=True); q0=sig.shift(1).loc[dt].rank(pct=True); turns.append((q-q0).abs().mean())
 a=np.array(ics); return a,np.mean(ns),np.nanmean(turns)
a,n,t=evaly(fwd)
print('factor=dispersion_gated_5d_reversal horizon=1d'); print('dates=%d avg_instruments=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%(len(a),n,np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),t))
for k in [250,500]:
 b=a[-k:]; print('recent%d IC=%.6f ICIR=%.6f hit=%.4f n=%d'%(k,np.mean(b),np.mean(b)/(np.std(b,ddof=1)+1e-12)*np.sqrt(len(b)),np.mean(b>0),len(b)))
for h in [5,10]:
 aa,_,_=evaly(px.pct_change(h).shift(-h)); print('h%d IC=%.6f ICIR=%.6f n=%d'%(h,np.mean(aa),np.mean(aa)/(np.std(aa,ddof=1)+1e-12)*np.sqrt(len(aa)),len(aa)))
