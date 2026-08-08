import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in files}
assets=sorted(D)
px=pd.concat({a:D[a]['close'] for a in assets},axis=1).sort_index()
ret=px.pct_change()
# residual momentum: 5d asset return less contemporaneous cross-sectional median, scaled by idiosyncratic 20d vol
csmed=ret.median(axis=1)
res=ret.sub(csmed,axis=0)
factor=(res.rolling(5).sum()/res.rolling(20).std()).shift(1)
# score high residual momentum; forward returns
out=[]
for h in [1,5,10,20]:
  fr=px.pct_change(h).shift(-h)
  ics=[]; dates=[]; ns=[]
  for dt in factor.index:
    x=factor.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
      ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
  a=np.array(ics); print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12),np.mean(a>0)))
# yearly daily
fr=ret.shift(-1); z=[]
for dt in factor.index:
 q=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(q)>=8:z.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
z=pd.DataFrame(z,columns=['date','ic']).set_index('date')
print('year',z.groupby(z.index.year).ic.mean().round(5).to_dict())
print('coverage',factor.notna().sum().mean()/15,'turnover',((factor.rank(axis=1,pct=True).diff().abs().sum(axis=1)/15).dropna().mean()))
print('range',px.index.min(),px.index.max())
