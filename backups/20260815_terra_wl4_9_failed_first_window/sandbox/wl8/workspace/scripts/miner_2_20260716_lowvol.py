import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); D[s]=x.set_index('date').close
P=pd.DataFrame(D).sort_index(); R=P.pct_change()
# Low realized volatility: lower trailing 20d vol ranks higher, a defensive cross-asset quality signal
F=-R.rolling(20).std()
rows={h:[] for h in [1,5,10]}; dates={h:[] for h in rows}; n={h:[] for h in rows}; turns=[]
for i in range(25,len(P)-10):
 for h in rows:
  a=F.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; z=pd.concat([a,y],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   rows[h].append(ic); dates[h].append(P.index[i]); n[h].append(len(z))
 # rank turnover among consecutive dates
 if i>25:
  a=F.iloc[i].rank(pct=True); b=F.iloc[i-1].rank(pct=True); turns.append((a-b).abs().mean())
for h in rows:
 x=np.array(rows[h]); print(h,'obs',len(x),'dates',len(set(dates[h])),'avgN',np.mean(n[h]),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'recent250',x[-250:].mean(),'decay')
print('turn',np.mean(turns),'coverage',F.notna().mean().mean())
