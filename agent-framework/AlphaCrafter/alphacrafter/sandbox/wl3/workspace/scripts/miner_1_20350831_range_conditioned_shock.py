import pandas as pd, numpy as np, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cutoff=pd.Timestamp('2035-08-30')
F={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); F[s]=d.set_index('date').sort_index()
idx=pd.DatetimeIndex(sorted(set().union(*[set(v.index[v.index<=cutoff]) for v in F.values()])))
o=pd.DataFrame({s:F[s].open.reindex(idx) for s in U}); c=pd.DataFrame({s:F[s].close.reindex(idx) for s in U}); h=pd.DataFrame({s:F[s].high.reindex(idx) for s in U}); l=pd.DataFrame({s:F[s].low.reindex(idx) for s in U})
c=c.ffill(); r=c.pct_change(); atr=((h-l).rolling(20,min_periods=10).mean()/c).shift(1)
shock=(o/c.shift(1)-1)/(atr+1e-12); basefac=(-shock).rolling(3,min_periods=3).mean()*(1+0.5*np.sign(-r.rolling(10,min_periods=10).sum()))
lo=c.rolling(60,min_periods=40).min(); hi=c.rolling(60,min_periods=40).max(); loc=(c-lo)/(hi-lo+1e-12)
# conditional: reversal signal strengthened at lower range locations, weakened at upper; avoids arbitrary continuous scale
factor=basefac*(0.7+0.6*(1-loc))
y=r.shift(-1); rows=[]
for dt in factor.index:
 x=factor.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].rank().corr(z[ok].rank()),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic.dropna()
print('candidate=range_conditioned_intraday_shock cutoff',cutoff.date()); print('dates',len(a),'instruments',15,'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('1d IC %.8f ICIR %.8f hit %.4f'%(ic.mean(),ic.mean()/(ic.std(ddof=1)+1e-12)*np.sqrt(len(ic)),(ic>0).mean()))
for hh in [3,5,10,20]:
 yy=c.pct_change(hh).shift(-hh); q=[]
 for dt in factor.index:
  x=factor.loc[dt]; z=yy.loc[dt]; ok=x.notna()&z.notna()
  if ok.sum()>=8:q.append(x[ok].rank().corr(z[ok].rank()))
 print('horizon',hh,'IC',np.nanmean(q),'n',len(q))
for name,x in [('early',ic.iloc[:len(ic)//3]),('middle',ic.iloc[len(ic)//3:2*len(ic)//3]),('recent',ic.iloc[2*len(ic)//3:]),('recent120',ic.tail(120))]: print(name,len(x),'IC',x.mean(),'ICIR',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',a.index.min().date(),a.index.max().date())
factor.reset_index().melt(id_vars='index',var_name='symbol',value_name='signal').rename(columns={'index':'date'}).dropna().to_csv('scripts/miner_1_20350831_range_conditioned_shock_signal.csv',index=False)
