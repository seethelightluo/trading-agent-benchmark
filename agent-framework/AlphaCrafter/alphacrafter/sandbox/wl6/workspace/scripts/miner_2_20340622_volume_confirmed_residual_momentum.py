import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); D[s]=x[['date','close','volume']].drop_duplicates('date').set_index('date').sort_index()
 except Exception as e: print('missing',s)
p=pd.DataFrame({s:D[s].close.astype(float) for s in D}).sort_index(); vv=pd.DataFrame({s:D[s].volume.astype(float) for s in D}).reindex(p.index)
r10=p.pct_change(10); rv=p.pct_change().rolling(30).std()*np.sqrt(30); res=r10.sub(r10.median(axis=1),axis=0)
vr=(vv/vv.rolling(60).median()).clip(.25,4); f=(res/(rv+1e-12))*(.75+.25*np.log(vr).clip(-1,1)); f=f.shift(1)
f.to_csv('scripts/miner_2_20340622_volume_confirmed_residual_momentum_signal.csv',index_label='date')
for H in [5,10,20,40]:
 fr=p.pct_change(H).shift(-H); vals=[]; ns=[]; turns=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  q=f.loc[dt].dropna().rank(pct=True)
  if prev is not None:
   c=q.index.intersection(prev.index)
   if len(c)>=8: turns.append(np.mean(abs(q[c]-prev[c])))
  if len(q):prev=q
 a=np.array(vals); print(H,'dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,5),'IC %.9f'%np.nanmean(a),'ICIR %.9f'%(np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(len(a))),'hit',round(np.mean(a>0),4),'turn',round(np.mean(turns),5))
fr=p.pct_change(20).shift(-20)
for label,years in [('2025-29',range(2025,2030)),('2030-32',range(2030,2033)),('2033-34',range(2033,2035))]:
 a=[]
 for dt in f.index:
  if dt.year not in years:continue
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(label,len(a),round(np.mean(a),6) if a else np.nan)
