import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try:
        x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); D[s]=x[['date','close']].drop_duplicates('date').set_index('date').sort_index().close.astype(float)
    except Exception as e: print('missing',s)
p=pd.DataFrame(D).sort_index(); r10=p.pct_change(10); vol=p.pct_change().rolling(30).std()*np.sqrt(252)
res=r10.sub(r10.median(axis=1),axis=0); f=-(res/vol).shift(1); f.to_csv('scripts/miner_2_20340608_medium_residual_reversal_signal.csv',index_label='date')
for H in [5,10,20,40]:
    fr=p.shift(-H)/p-1; vals=[]; ns=[]; turns=[]; prev=None
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
        q=f.loc[dt].dropna().rank(pct=True)
        if prev is not None:
            common=q.index.intersection(prev.index)
            if len(common)>=8: turns.append(np.mean(abs(q[common]-prev[common])))
        if len(q): prev=q
    ar=np.array(vals); print(H,'dates',len(ar),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,5),'IC %.9f'%np.nanmean(ar),'ICIR %.9f'%(np.nanmean(ar)/np.nanstd(ar,ddof=1)*np.sqrt(len(ar))),'hit',round(np.mean(ar>0),4),'turn',round(np.mean(turns),5))
H=20; fr=p.shift(-H)/p-1
for label,years in [('2025-29',range(2025,2030)),('2030-32',range(2030,2033)),('2033-34',range(2033,2035))]:
 vals=[]
 for dt in f.index:
  if dt.year not in years: continue
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(label,len(vals),round(np.mean(vals),6) if vals else np.nan)
