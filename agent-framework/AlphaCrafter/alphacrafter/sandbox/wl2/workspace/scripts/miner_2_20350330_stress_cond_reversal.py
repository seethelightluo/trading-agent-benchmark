import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end='2035-03-30'
px={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d[d.date<=pd.Timestamp(end)].set_index('date').sort_index()
        px[s]=d['close'].astype(float)
P=pd.concat(px,axis=1).sort_index()
R=P.pct_change()
# cross-asset daily dispersion, lagged and smoothed; high dispersion = stress/opportunity regime
cs_disp=R.std(axis=1,ddof=1)
disp_rank=cs_disp.rolling(252,min_periods=126).rank(pct=True)
# factor: lagged 5d reversal, risk scaled, activated at elevated dispersion (rank known at t-1)
rev=-(P.pct_change(5).shift(1)) / (R.rolling(20,min_periods=15).std().shift(1)*np.sqrt(20))
# soft conditional activation avoids all-zero ties: multiplier 0.25 to 1 based on prior dispersion rank
mult=(0.25+0.75*disp_rank.shift(1)).clip(0.25,1.0)
F=rev.mul(mult,axis=0)
# forward close-to-close returns from signal date
for h in [5,10,20,40]:
    fr=P.shift(-h)/P-1
    vals=[]; dates=[]; ns=[]
    for dt in F.index:
        x=F.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
        if ok.sum()>=8:
            vals.append(x[ok].corr(y[ok],method='spearman')); dates.append(dt); ns.append(int(ok.sum()))
    a=np.asarray(vals,float); a=a[np.isfinite(a)]
    ic=a.mean(); icir=ic/a.std(ddof=1) if len(a)>1 else np.nan
    print(f'h{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={ic:.8f} ICIR={icir:.8f} hit={np.mean(a>0):.4f}')
# coverage and turnover on valid cross sections
valid=F.notna().sum(axis=1); print('rows',len(P),'dates',P.index.min().date(),P.index.max().date(),'coverage',valid.mean()/len(U))
# rank turnover proxy
r=F.rank(axis=1,pct=True); turn=r.diff().abs().mean(axis=1).mean(); print('turnover_proxy',turn)
# regime split h10
fr=P.shift(-10)/P-1; vals=[]
for dt in F.index:
 ok=F.loc[dt].notna()&fr.loc[dt].notna()
 if ok.sum()>=8: vals.append((dt,F.loc[dt,ok].corr(fr.loc[dt,ok],method='spearman')))
z=pd.DataFrame(vals,columns=['date','ic']).set_index('date')
for name,sub in [('pre2025',z[z.index<'2025-01-01']),('2025+',z[z.index>='2025-01-01']),('2030+',z[z.index>='2030-01-01'])]: print(name,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1) if len(sub)>1 else np.nan)
# artifact for audit
out=F.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/miner_2_20350330_stress_cond_reversal_signal.csv',index=False)
