import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cutoff=pd.Timestamp('2027-09-22')
prices={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); d=d[d.date<=cutoff].sort_values('date').drop_duplicates('date'); prices[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(prices).sort_index(); r=px.pct_change(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20); f=(-px.pct_change(5)/vol).replace([np.inf,-np.inf],np.nan)
rows=[]
for dt,row in f.iterrows():
 for s,v in row.items():
  if pd.notna(v): rows.append({'date':dt.strftime('%Y-%m-%d'),'symbol':s,'value':float(v)})
pd.DataFrame(rows).to_csv('factors/miner_2_20270923_volnorm_reversal_signal.csv',index=False)
for h in [1,5,10]:
 vals=[]; dates=[]; ns=[]
 for dt in px.index:
  nxt=px.index[px.index>dt]
  if len(nxt)<h: continue
  y=px.loc[nxt[h-1]]/px.loc[dt]-1; z=pd.concat([f.loc[dt],y],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 a=np.array(vals); mean=np.nanmean(a); icir=mean/np.nanstd(a,ddof=1)*np.sqrt(252)
 print(f'h={h} dates={len(a)} avg_names={np.mean(ns):.2f} IC={mean:.6f} ICIR={icir:.6f} hit={np.mean(a>0):.3f}')
 for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01','2027-09-22')]:
  q=np.array([x for x,d in zip(vals,dates) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)]); print(' ',label,len(q),f'{np.nanmean(q):.6f}' if len(q) else 'NA')
valid=f.notna().sum(axis=1); print('coverage',valid.mean()/len(U),'mean_valid',valid.mean()); print('turnover_proxy',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
