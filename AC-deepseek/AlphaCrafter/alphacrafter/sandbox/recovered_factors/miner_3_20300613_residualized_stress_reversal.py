import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame(px).sort_index().astype(float); ret=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# point-in-time stress gate: two completed observations, q80 trailing 120
raw=v>v.rolling(120,min_periods=60).quantile(.8); gate=(raw.shift(1)&raw.shift(2)).astype(bool)
base=-(p/p.shift(20)-1); vol=ret.rolling(20,min_periods=15).std(); trend=(p/p.shift(5)-1)+(p/p.shift(20)-1)/4
# Residualize gated reversal cross-sectionally against raw reversal, volatility, and trend; only active stress dates.
x=base.where(gate); sig=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
for d in p.index:
    z=pd.DataFrame({'x':x.loc[d],'b':base.loc[d],'v':vol.loc[d],'t':trend.loc[d]}).dropna()
    if len(z)>=8:
        A=np.column_stack([np.ones(len(z)),z[['b','v','t']].rank(pct=True).values])
        sig.loc[d,z.index]=z.x.values-A@np.linalg.lstsq(A,z.x.values,rcond=None)[0]
# forward returns, cross-sectional IC
print('candidate=residualized_lagged_vix_q80_reversal20')
print('dates',len(p),'instruments',len(p.columns),'active_dates',int(gate.sum()),'coverage',sig.notna().sum().sum()/sig.size)
for h in [1,5,10,20]:
  vals=[]; ns=[]; ds=[]
  f=p.shift(-h)/p-1
  for d in p.index:
    z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
    if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(d)
  a=np.asarray(vals); print('horizon',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
# active signal turnover and diagnostics
r=sig.rank(axis=1,pct=True); print('turnover_proxy',round(r.diff().abs().mean(axis=1).dropna().mean(),6))
# correlations with components on common signal cells
for name,c in [('base',base),('vol',vol),('trend',trend)]:
 z=pd.concat([sig.stack().rename('s'),c.stack().rename('c')],axis=1).dropna(); print('corr',name,round(spearmanr(z.s,z.c).statistic,6),'cells',len(z))
# regime 10d
f=p.shift(-10)/p-1
for label,mask in [('2020-23',p.index<'2024-01-01'),('2024-27',(p.index>='2024-01-01')&(p.index<'2028-01-01')),('2028+',p.index>='2028-01-01'),('latest120',pd.Series(False,index=p.index).set_axis(p.index).index>=p.index[-120])]:
 vals=[]
 for d in p.index[mask]:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(vals); print('regime',label,'dates',len(a),'IC',round(a.mean(),6) if len(a) else None,'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
