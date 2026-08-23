import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in A:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].astype(float)
 px[s]=d.sort_index()
p=pd.concat(px,axis=1).sort_index().loc[:'2033-03-02']
r=p.pct_change(); mom=p.pct_change(20)
down=r.where(r<0).pow(2).rolling(30,min_periods=15).mean().pow(.5)
raw=mom.div(down.replace(0,np.nan))
breadth=(mom>0).mean(axis=1)
# hysteresis: only flip at broad agreement; otherwise retain prior state
state=[]; cur=1
for b in breadth:
 if pd.notna(b):
  if b>=.65: cur=1
  elif b<=.35: cur=-1
 state.append(cur)
sig=raw.mul(pd.Series(state,index=p.index),axis=0)
def calc(h):
 f=p.shift(-h).div(p)-1; out=[];ns=[];dates=[]
 for i in range(len(p)-h):
  z=pd.concat([sig.iloc[i].rename('x'),f.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:
   out.append(spearmanr(z.x,z.y).statistic); ns.append(len(z)); dates.append(p.index[i])
 x=np.asarray(out); ser=pd.Series(x,index=pd.DatetimeIndex(dates))
 return len(x),np.mean(ns),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(np.asarray(ns)/15),ser
print('universe',len(A),'data_dates',len(p),'signal_coverage',round(float(sig.notna().stack().mean()),4))
for h in [5,10,20,40]:
 n,an,ic,ir,hit,cov,ser=calc(h)
 print('horizon',h,'dates',n,'avg_n',round(an,2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4),'coverage',round(cov,4))
 print('regimes',ser.groupby(ser.index.year).mean().round(5).to_dict())
u=[]
for i in range(1,len(sig)):
 z=pd.concat([sig.iloc[i-1].rank(pct=True),sig.iloc[i].rank(pct=True)],axis=1).dropna()
 if len(z): u.append(np.mean(abs(z.iloc[:,1]-z.iloc[:,0])))
print('turnover',round(float(np.mean(u)),6),'max_abs_library_correlation','not_computed')
