import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
assets=sorted(D)
close=pd.concat({a:D[a]['close'] for a in assets},axis=1).sort_index()
vol=pd.concat({a:D[a]['volume'] for a in assets},axis=1).sort_index()
ret=close.pct_change()
# Volume surprise orthogonalized to contemporaneous absolute return: isolate participation
# that is not merely a consequence of a large price move. Both components are lagged.
vs=(vol/(vol.rolling(20,min_periods=10).median()+1e-12)).clip(upper=5)
vr=np.log1p(vs.rolling(3,min_periods=3).mean())
move=ret.abs().rolling(3,min_periods=3).mean()
def cs_resid(y,x):
    out=y* np.nan
    for dt in y.index:
        q=pd.concat([y.loc[dt],x.loc[dt]],axis=1).dropna()
        if len(q)>=8 and q.iloc[:,1].std()>1e-12:
            b=np.cov(q.iloc[:,0],q.iloc[:,1],ddof=1)[0,1]/q.iloc[:,1].var(ddof=1)
            a=q.iloc[:,0].mean()-b*q.iloc[:,1].mean()
            out.loc[dt,q.index]=q.iloc[:,0]-a-b*q.iloc[:,1]
    return out
v_res=cs_resid(vr,move)
# Residual participation weighted short-term reversal; lag avoids using forward information.
factor=-(ret.rolling(3,min_periods=3).sum())*v_res
factor=factor.shift(1)
for h in [1,5,10,20]:
 fr=close.pct_change(h).shift(-h); ics=[]; ns=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(ics); print('H',h,'dates',len(a),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()))
fr=ret.shift(-1); rows=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
z=pd.DataFrame(rows,columns=['date','ic']).set_index('date')
print('years',z.groupby(z.index.year).ic.mean().round(5).to_dict())
r=factor.rank(axis=1,pct=True)
print('coverage %.4f avg_valid %.2f dates %d turnover %.5f'%(factor.notna().sum().sum()/factor.size,factor.notna().sum(axis=1).mean(),len(z),r.diff().abs().sum(axis=1).div(15).dropna().mean()))
print('regimes',[(y,len(g),round(g.ic.mean(),5),round((g.ic>0).mean(),4)) for y,g in z.groupby(z.index.year)])
