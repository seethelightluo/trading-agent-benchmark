import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); D[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index().ffill(); R=P.pct_change();
# Regime-adaptive medium trend: direction follows 10d cross-sectional breadth; scale by 30d risk.
ret10= P.pct_change(10); vol30=R.rolling(30,min_periods=20).std(); breadth=(ret10>0).mean(axis=1).shift(1)
reg=np.where(breadth>=0.5,1.0,-1.0)
F=ret10.shift(1).div(vol30.shift(1)*np.sqrt(10)+1e-12).mul(reg,axis=0)
rows=[]
for h in [1,3,5,10,20]:
  ics=[]; cov=[]; turns=[]
  for i in range(45,len(P)-h):
   x=F.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8:
    q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(q): ics.append(q); cov.append(len(z)/len(U))
    if i>45:
     old=F.iloc[i-1].reindex(z.index)
     turns.append((x.rank()!=old.rank()).mean())
  a=pd.Series(ics); rows.append((h,len(a),a.mean(),a.mean()/a.std(),(a>0).mean(),np.mean(cov),np.mean(turns)))
print('dates',len(P),'instruments',len(D),'last',P.index.max())
print('h,n,IC,ICIR,hit,cov,turn')
for x in rows: print(x)
F.index.name='date'; F.to_csv('scripts/miner_3_20350914_regime_adaptive_trend_signal.csv')
