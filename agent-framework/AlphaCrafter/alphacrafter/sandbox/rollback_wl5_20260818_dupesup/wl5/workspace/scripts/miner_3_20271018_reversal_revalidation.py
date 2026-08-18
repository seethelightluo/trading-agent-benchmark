import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for s in U}
def evaluate(lb,lo,hi):
    rows=[]
    for s,x in D.items():
        r=x.close.pct_change(lb)
        for i in range(len(x)-1):
            if lo<=x.index[i]<=hi and pd.notna(r.iloc[i]): rows.append((x.index[i],s,float(r.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
    a=pd.DataFrame(rows,columns=['date','s','r','y']); z=[]
    for d,g in a.groupby('date'):
        if len(g)>=8 and g.r.nunique()>1 and g.y.nunique()>1: z.append(spearmanr(-g.r,g.y).statistic)
    z=np.asarray(z); return len(z),a.s.nunique(),a.groupby('date').size().mean(),np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0)
for lb in [3,5,10,20]:
 for label,lo,hi in [('full','2020-02-01','2027-10-18'),('recent','2026-07-16','2027-10-18')]:
  n,ins,m,ic,ir,hit=evaluate(lb,pd.Timestamp(lo),pd.Timestamp(hi)); print(lb,label,'dates',n,'ins',ins,'meanN',round(m,2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4))
