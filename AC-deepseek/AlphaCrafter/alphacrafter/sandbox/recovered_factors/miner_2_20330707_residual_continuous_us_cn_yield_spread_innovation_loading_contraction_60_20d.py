"""One candidate: residual continuous US--China yield-spread innovation loading contraction, 60d vs 20d.
The observable driver is the trailing-standardized change in the US10Y minus CN10Y yield spread.
The score is an asset's residual-return beta to this rate-differential innovation, structural minus recent.
Validation cutoff is the last completed day before 2033-07-07.
"""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2033-07-06')
def load(a):
    return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:END]
p=pd.concat({a:load(a) for a in A},axis=1).sort_index().ffill()
r=p.pct_change(); m=r.mean(axis=1)
def rb(x,y,w,n):
    return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/(y.rolling(w,min_periods=n).var()+1e-12) for a in A})
e=r-rb(r,m,60,40).mul(m,axis=0)
# The two yield series are tradable prices in the benchmark but their relative daily movement
# is still a directly observable cross-market rate-differential state measure.
spread=r['US10Y']-r['CN10Y']
driver=((spread-spread.rolling(60,min_periods=40).mean())/(spread.rolling(60,min_periods=40).std()+1e-12)).clip(-5,5)
f=rb(e,driver,60,42)-rb(e,driver,20,14)
f.to_pickle('scripts/miner_2_20330707_us_cn_yield_spread_innovation_signal.pkl')
print('FACTOR residual_continuous_us_cn_yield_spread_innovation_loading_contraction_60_20d','VALIDATION_END',END.date(),'PANEL',p.index.min().date(),p.index.max().date(),'ASSETS',len(A),'VALID_CELLS',int(f.notna().sum().sum()),'COVERAGE',round(float(f.notna().mean().mean()),6),'DRIVER_COVERAGE',round(float(driver.notna().mean()),6))
ics={}; metrics={}
for h in [1,5,10,20]:
    fw=p.shift(-h).div(p).sub(1); vals=[]; ns=[]
    for t in p.index:
        q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
        if len(q)>=8 and q.f.nunique()>1:
            v=q.f.corr(q.y,method='spearman')
            if pd.notna(v): vals.append((t,v)); ns.append(len(q))
    s=pd.Series(dict(vals),dtype=float); ics[h]=s; sd=s.std(ddof=1)
    metrics[h]=(s.mean(),s.mean()/sd,(s>0).mean(),len(s),np.mean(ns),sd/np.sqrt(len(s)))
    print('H',h,'IC %.6f ICIR %.6f HIT %.6f DATES %d MEAN_N %.2f SE %.6f'%metrics[h])
for name,mask in [('2020_2024',ics[10].index<pd.Timestamp('2025-01-01')),('2025_2026',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
    s=ics[10][mask]; print('REGIME10',name,'DATES',len(s),'IC %.6f ICIR %.6f HIT %.6f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
    q=ranks.iloc[[i-1,i]].T.dropna()
    if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('RANK_TURNOVER %.6f TURNOVER_DATES %d'%(np.mean(turns),len(turns)))
print('DECAY',json.dumps({str(h):{'ic':round(float(metrics[h][0]),6),'icir':round(float(metrics[h][1]),6),'dates':metrics[h][3]} for h in metrics}))
print('LIBRARY_CORRELATION_STATUS NOT_COMPUTED_ALL_30_CURRENT_EFFECTIVE_SIGNAL_DEFINITIONS_REQUIRE_RECONSTRUCTION')
