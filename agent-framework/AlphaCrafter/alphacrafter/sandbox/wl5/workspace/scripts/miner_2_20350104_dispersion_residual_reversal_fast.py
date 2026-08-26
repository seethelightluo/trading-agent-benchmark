import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2200)
 if d is not None and len(d)>=180: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); res=r.sub(r.mean(axis=1),axis=0)
base=(-res.rolling(10,min_periods=8).sum()/(res.rolling(40,min_periods=25).std()*np.sqrt(10)+1e-12)).clip(-8,8)
disp=res.std(axis=1).rolling(20,min_periods=12).mean(); norm=disp/disp.rolling(252,min_periods=80).median()
S=base*(.75+.5*norm.clip(.5,1.5)); Q=P.shift(-10)/P-1
# rowwise Spearman correlation, requiring >=8
sr=S.rank(axis=1); qr=Q.rank(axis=1); ok=sr.notna()&qr.notna(); n=ok.sum(axis=1)
x=sr.where(ok); y=qr.where(ok); xm=x.mean(axis=1); ym=y.mean(axis=1)
num=((x.sub(xm,axis=0))*(y.sub(ym,axis=0))).sum(axis=1); den=np.sqrt(((x.sub(xm,axis=0)**2).sum(axis=1))*((y.sub(ym,axis=0)**2).sum(axis=1)))
ic=(num/den).where(n>=8).dropna(); print('assets',len(P.columns),'rows',len(P),'dates',len(ic),'start',ic.index[0].date(),'end',ic.index[-1].date(),'mean_n',round(n[ic.index].mean(),3),'coverage',round(n[ic.index].mean()/15,6)); print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),6))
for x0,x1 in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2035-01-03')]:
 z=ic.loc[x0:x1]; print('regime',x0,len(z),round(z.mean(),6) if len(z) else None)
print('turnover',round(S.rank(axis=1).diff().abs().mean().mean(),6))
for h in [5,20]:
 qh=P.shift(-h)/P-1; yy=qh.rank(axis=1); ok2=sr.notna()&yy.notna(); nn=ok2.sum(axis=1); xx=sr.where(ok2); yy=yy.where(ok2); mx=xx.mean(axis=1); my=yy.mean(axis=1); cc=(((xx.sub(mx,axis=0))*(yy.sub(my,axis=0))).sum(axis=1)/np.sqrt((xx.sub(mx,axis=0)**2).sum(axis=1)*(yy.sub(my,axis=0)**2).sum(axis=1))).where(nn>=8).dropna(); print('decay',h,round(cc.mean(),6),'dates',len(cc))
pd.DataFrame([(dt,s,float(S.loc[dt,s])) for dt in S.index for s in S.columns if pd.notna(S.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20350104_dispersion_residual_reversal_signal.csv',index=False)
