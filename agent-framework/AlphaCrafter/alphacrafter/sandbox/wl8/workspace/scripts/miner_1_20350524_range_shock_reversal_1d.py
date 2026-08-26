import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p):continue
 d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); d=d.sort_values('date'); r=d.close.pct_change(); z=(d.high-d.low)/d.close.shift(1); sh=z/(z.rolling(30,min_periods=20).median()+1e-12); v=r.rolling(30,min_periods=20).std(); f=-(r.shift(1))*sh.shift(1)/(v.shift(1)+1e-12); y=d.close.shift(-1)/d.close-1; rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor':f,'fwd':y}).dropna())
p=pd.concat(rows,ignore_index=True); out=[]; sig=[]
for dt,g in p.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1:out.append((dt,spearmanr(g.factor,g.fwd).statistic)); sig.extend(g[['date','symbol','factor']].to_dict('records'))
i=pd.DataFrame(out,columns=['date','ic']); m=i.ic.mean(); ir=m/i.ic.std(ddof=1)*np.sqrt(252); x=p.sort_values(['symbol','date']); x['rank']=x.groupby('date')['factor'].rank(pct=True); x['delta']=x.groupby('symbol')['rank'].diff().abs(); print({'dates':len(i),'instruments':len(U),'avg_instruments':p.groupby('date').size().mean(),'mean_daily_paper_ic':m,'daily_paper_icir':ir,'hit_ratio':(i.ic>0).mean(),'turnover_proxy':x.delta.mean()}); i.to_csv('scripts/miner_1_20350524_range_shock_reversal_1d_ic.csv',index=False); pd.DataFrame(sig).to_csv('scripts/miner_1_20350524_range_shock_reversal_1d_signal.csv',index=False)
