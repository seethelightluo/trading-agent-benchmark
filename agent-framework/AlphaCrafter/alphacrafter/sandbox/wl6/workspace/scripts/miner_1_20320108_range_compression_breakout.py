import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-01-07')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut]
 px[s]=d
p=pd.concat(px,axis=1).sort_index(); r=p.pct_change()
# Factor: recent return scaled by short vol, amplified when short vol is compressed versus medium vol
ret10=p.pct_change(10); v10=r.rolling(10).std(); v40=r.rolling(40).std()
f=(ret10/v10.replace(0,np.nan))*(v40/v10.replace(0,np.nan)).clip(0.25,4)
rows=[]
for h in [5,10,20]:
 ic=[]; dates=[]; n=[]
 fr=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt);n.append(len(z))
 a=np.array(ic); print('H',h,'IC %.8f ICIR %.5f hit %.4f dates %d avgN %.3f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),len(a),np.mean(n)))
 if h==5:
  # rank turnover from adjacent valid signals
  ranks=f.rank(axis=1,pct=True); tv=(ranks.diff().abs().mean(axis=1)).dropna().mean(); print('turnover_proxy %.8f coverage %.6f'%(tv,f.notna().mean().mean()))
# yearly 5d
fr=p.shift(-5)/p-1
for y in range(2020,2032):
 vals=[]
 for dt in f.index[f.index.year==y]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 if vals: print('Y',y,'n',len(vals),'ic %.6f'%(np.mean(vals)))
