import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in syms:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); prices[s]=d.set_index('date').close
p=pd.DataFrame(prices).sort_index(); r=p.pct_change(); ret20=p.pct_change(20); vol30=r.rolling(30).std()
breadth=(ret20<0).mean(axis=1); raw=-ret20/vol30
res=raw.sub(raw.median(axis=1),axis=0)
f=res*(1+0.8*(breadth-0.5))
rows=[]
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1
 for dt in f.index[:-h]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 out=pd.DataFrame(rows,columns=['date','n','ic']); ic=out.ic.mean(); sd=out.ic.std(ddof=1); icir=ic/sd*np.sqrt(252)
 recent=out.tail(120).ic; ric=recent.mean(); ricir=ric/recent.std(ddof=1)*np.sqrt(252)
 print('H',h,'dates',len(out),'avgN',out.n.mean(),'IC',ic,'ICIR',icir,'hit',(out.ic>0).mean(),'recent120',ric,ricir,'coverage',out.n.mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
 if h==20:
  os.makedirs('scripts/artifacts',exist_ok=True); f.to_csv('scripts/artifacts/miner_2_20331027_breadth_conditioned_reversal_signal.csv'); out.to_csv('scripts/artifacts/miner_2_20331027_breadth_conditioned_reversal_ic.csv',index=False)
