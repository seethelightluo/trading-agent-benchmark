import pandas as pd,numpy as np,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; start=pd.Timestamp('2026-07-16'); end=pd.Timestamp('2033-04-17'); D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc[start:end]; r=px.pct_change()
# Downside-risk-adjusted medium momentum: 20d return divided by downside deviation over 40d, lagged one completed session.
down=r.clip(upper=0).rolling(40).std(); sig=(px/px.shift(20)-1)/(down+1e-8); sig=sig.shift(1)
fout=px.shift(-10)/px-1
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(v): vals.append(v);ns.append(len(q))
 a=np.array(vals); print('H',h,'dates',len(a),'assets_avg',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
a=[]
for dt in px.index:
 q=pd.concat([sig.loc[dt],fout.loc[dt]],axis=1).dropna()
 if len(q)>=8:
  v=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
  if np.isfinite(v): a.append(v)
a=np.array(a); n=len(a); print('thirds',[round(x,6) for x in np.array_split(a,3) if len(x) for x in [np.mean(x)]])
print('coverage',round(sig.notna().mean().mean(),4),'n_assets',len(D),'n_dates',len(px)); sig.to_csv('scripts/miner_2_20330418_downside_momentum_signal.csv')
