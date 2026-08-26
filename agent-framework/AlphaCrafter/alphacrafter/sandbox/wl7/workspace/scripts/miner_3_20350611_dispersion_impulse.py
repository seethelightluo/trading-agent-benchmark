import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-06-10']
r=C.pct_change(); vol=r.rolling(20,min_periods=15).std(); disp=vol.median(axis=1)
# Candidate: cross-sectional residual short-term reversal, activated by a dispersion impulse.
# Impulse requires dispersion above its trailing 60d median and rising versus 5d ago; all inputs lagged one day.
raw=-(C/C.shift(5)-1); raw=raw.sub(raw.median(axis=1),axis=0); sig=(raw/vol)
gate=(disp>disp.rolling(60,min_periods=40).median()) & (disp>disp.shift(5))
f=sig.where(gate).shift(1)
y=C.shift(-20)/C-1
a=[]; ds=[]; ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);ns.append(len(z))
a=np.asarray(a); ds=np.asarray(ds)
print('candidate=dispersion_impulse_residual_reversal5_h20')
print('IC %.8f ICIR %.8f dates %d avgN %.2f hit %.4f coverage %.4f active_dates %.4f'%(a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),len(a),np.mean(ns),np.mean(a>0),f.notna().sum(axis=1).mean()/15,np.mean(gate)))
for st,en in [('2027','2030-12-31'),('2031','2034-12-31'),('2035','2035-06-10')]:
 q=a[(ds>=pd.Timestamp(st))&(ds<=pd.Timestamp(en))]; print(st,len(q),('%.8f'%q.mean()) if len(q) else 'nan')
# decay
for h in [1,5,10,20]:
 yy=C.shift(-h)/C-1; aa=[]
 for d in f.index:
  z=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 aa=np.asarray(aa); print('H',h,'IC %.8f ICIR %.8f dates %d'%(aa.mean(),aa.mean()/aa.std(ddof=1)*np.sqrt(252),len(aa)))
f.to_csv('scripts/miner_3_20350611_dispersion_impulse_signal.csv',index_label='date')
