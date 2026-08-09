import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
watch=[os.path.basename(f)[:-4] for f in files]
px={}
for f in files:
 d=pd.read_csv(f); px[os.path.basename(f)[:-4]]=d.set_index('date')['close']
P=pd.DataFrame(px).sort_index()
r=np.log(P).diff()
# downside-adjusted 20-session momentum: return divided by recent downside deviation
# signal at t uses through t only; tested against t+1 return
mom=np.log(P/P.shift(20))
down=r.where(r<0,0).rolling(40,min_periods=20).std()
sig=mom/(down*np.sqrt(252)+1e-12)
ics=[]; ns=[]; dates=[]; turnovers=[]; prev=None
for i in range(1,len(P)-1):
 x=sig.iloc[i]; y=r.iloc[i+1]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum()); dates.append(P.index[i])
  ranks=x.rank(pct=True); turnovers.append(np.mean(abs(ranks-(prev if prev is not None else ranks)).dropna()))
  prev=ranks
z=np.array(ics); print('candidate downside_adjusted_momentum_20_40')
print('dates',len(z),'meanN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',np.nanmean(z),'ICIR',np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),'hit',np.mean(z>0),'turnover',np.mean(turnovers))
for a,b in [('2020','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=z[np.array([(d>=a+'-01-01')&(d<=b+'-12-31') for d in dates])]; print(a,b,len(q),np.mean(q) if len(q) else np.nan,np.mean(q)/(np.std(q,ddof=1)+1e-12) if len(q)>1 else np.nan)
# decay
for h in [1,5,10,20]:
 yy=np.log(P.shift(-h)/P)
 q=[]
 for i in range(1,len(P)-h):
  ok=sig.iloc[i].notna()&yy.iloc[i].notna()
  if ok.sum()>=8:q.append(spearmanr(sig.iloc[i][ok],yy.iloc[i][ok]).statistic)
 q=np.array(q); print('H',h,'n',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/(np.std(q,ddof=1)+1e-12))
print('instruments',len(P.columns),P.columns.tolist())
