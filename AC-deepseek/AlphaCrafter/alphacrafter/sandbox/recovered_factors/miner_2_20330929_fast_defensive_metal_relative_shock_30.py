"""Fast, point-in-time validation of one factor: defensive-metal stress transmission.
Does not reconstruct the active-library signals: therefore output is research evidence,
not admission evidence.  All inputs end at 2033-09-28 (last completed session)."""
import numpy as np, pandas as pd
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2033-09-28')
def load(a):
 d=pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date')
 return d['close'].rename(a)
P=pd.concat([load(a) for a in A],axis=1).sort_index().loc[:END].ffill()
R=P.pct_change(); M=R.median(axis=1)
# Shock is known at t; 30-day conditional beta uses only rows through t.
S=R['XAU']-R['COPPER']; down=M<0
# rolling conditional beta using masked inputs (min 8 relevant obs); all operations vectorized by asset
X=pd.DataFrame(np.repeat(S.to_numpy()[:,None],len(A),axis=1),index=R.index,columns=A).where(np.repeat(down.to_numpy()[:,None],len(A),axis=1))
Y=R.where(np.repeat(down.to_numpy()[:,None],len(A),axis=1))
def rb(x,y):
 return x.rolling(30,min_periods=8).cov(y).div(x.rolling(30,min_periods=8).var())
bdown=rb(X,Y)
X2=pd.DataFrame(np.repeat(S.to_numpy()[:,None],len(A),axis=1),index=R.index,columns=A).where(np.repeat((~down).to_numpy()[:,None],len(A),axis=1))
Y2=R.where(np.repeat((~down).to_numpy()[:,None],len(A),axis=1))
bup=rb(X2,Y2)
raw=-(bdown-bup)
vol=R.rolling(20,min_periods=12).std()
# mean pairwise correlation, directly equivalent to peer crowding proxy
peer=pd.DataFrame(index=R.index,columns=A,dtype=float)
for a in A:
 peer[a]=R[a].rolling(20,min_periods=12).corr(R.drop(columns=a).mean(axis=1))
# downside beta asymmetry and trend
mdown=M.where(M<0); mup=M.where(M>=0)
db=R.rolling(30,min_periods=15).cov(mdown).div(mdown.rolling(30,min_periods=15).var()) - R.rolling(30,min_periods=15).cov(mup).div(mup.rolling(30,min_periods=15).var())
trend=P.pct_change(20)
# Cross-sectional residual each day with intercept; preserve only >=8 observations
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for t in P.index:
 y=raw.loc[t]; z=pd.concat([vol.loc[t],peer.loc[t],db.loc[t],trend.loc[t]],axis=1)
 ok=y.notna() & z.notna().all(axis=1)
 if ok.sum()>=8:
  xx=np.column_stack([np.ones(ok.sum()),z.loc[ok].to_numpy()])
  F.loc[t,ok]=y.loc[ok]-xx@np.linalg.lstsq(xx,y.loc[ok],rcond=None)[0]
def sp(x,y): return x.corr(y,method='spearman') if len(x)>=8 else np.nan
print('FACTOR inverse_defensive_metal_relative_shock_transmission_residual_30')
print('visible_through',END.date(),'observations',len(P),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fr=P.shift(-h).div(P)-1; vals=[]; ns=[]
 for t in P.index:
  d=pd.concat([F.loc[t],fr.loc[t]],axis=1).dropna()
  if len(d)>=8: vals.append(sp(d.iloc[:,0],d.iloc[:,1])); ns.append(len(d))
 x=pd.Series(vals); ics[h]=x
 print('H',h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'dates',len(x),'mean_n',round(np.mean(ns),2))
# turnover and coverage
rank=F.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1)
print('coverage',round(F.notna().mean().mean(),5),'valid_cells',int(F.notna().sum().sum()),'daily_rank_turnover',round(turn.mean(),6))
# two broad historical partitions, selected 20d
x=ics[20]; dates=[]
for t in P.index:
 d=pd.concat([F.loc[t],(P.shift(-20).div(P)-1).loc[t]],axis=1).dropna()
 if len(d)>=8: dates.append(t)
x.index=dates
for name, z in [('2020_2026',x[x.index<'2027-01-01']),('2027_2033',x[x.index>='2027-01-01'])]:
 print('REGIME',name,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'dates',len(z))
print('NOVELTY NOT COMPUTED: full admitted-library signal reconstruction required; candidate is ineligible for persistence.')
