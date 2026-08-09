"""One close-only candidate: return-sign alternation reversal, tested through 2029-01-24.
The score is the 20-observation rate at which consecutive non-zero daily return signs
flip. High alternation represents a choppy mean-reverting path; it is tested as a
cross-asset reversal-quality signal using only closes known at the signal date.
"""
import ast,glob,json,os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-01-24')
def get(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,'close'].astype(float)
P=pd.DataFrame({a:get(a) for a in A}).sort_index(); r=P.pct_change()
# A high sign-flip rate is a magnitude-free measure of choppiness; zeros are unavailable rather than classified.
s=np.sign(r).replace(0,np.nan)
flip=(s*s.shift(1)<0).astype(float).where(s.notna()&s.shift(1).notna())
F=flip.rolling(20,min_periods=15).mean()
def metrics(h):
 fw=P.shift(-h)/P-1; vals=[]; n=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1: vals.append((d,float(spearmanr(z.f,z.r).statistic))); n.append(len(z))
 ss=pd.Series(dict(vals),dtype=float); sd=ss.std(ddof=1)
 return ss,{'daily_paper_ic':float(ss.mean()),'daily_paper_icir':float(ss.mean()/sd),'ic_hit_ratio':float((ss>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ss))),'ic_dates':len(ss),'mean_valid_instruments':float(np.mean(n))}
for h in [1,5,10,20]:
 ss,m=metrics(h); print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==5:
  for lab,mask in [('2020_2021',ss.index.year<=2021),('2022_2023',ss.index.year.isin([2022,2023])),('2024_2025',ss.index.year.isin([2024,2025])),('2026_2028',ss.index.year.isin([2026,2027,2028])),('2029',ss.index.year==2029)]:
   q=ss[mask]; print('REGIME_5D',lab,json.dumps({'dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,'hit':float((q>0).mean()) if len(q) else None}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('PANEL',json.dumps({'signal_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'implied_turnover':float(1-np.mean(st))}))
out='scripts/miner_2_20290125_return_sign_alternation_reversal_20obs_signal.pkl';F.to_pickle(out)
src=open('scripts/miner_2_20281102_return_sign_entropy_continuation_20obs.py').read();tree=ast.parse(src);alias={}
for node in ast.walk(tree):
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='alias' for t in node.targets): alias=ast.literal_eval(node.value)
rows=[];missing=[]
for path in glob.glob('factors/*.json'):
 if path.endswith('.bak'):continue
 rec=json.load(open(path))
 if rec.get('validation',{}).get('status')!='EFFECTIVE':continue
 fid=rec['factor_id'];fn=alias.get(fid)
 if not fn or not os.path.exists('scripts/'+fn):missing.append(fid);continue
 L=pd.read_pickle('scripts/'+fn);x,y=F.align(L,join='inner',axis=0);z=pd.DataFrame({'x':x.stack(),'y':y.stack()}).dropna();rho=float(spearmanr(z.x,z.y).statistic) if len(z)>2 else float('nan');rows.append((fid,len(z),rho));print('LIBRARY_CORR',fid,len(z),rho)
print('LIBRARY_MISSING',json.dumps(missing))
if not missing and rows:
 b=max(rows,key=lambda q:abs(q[2]));print('LIBRARY_MAX',json.dumps({'factor':b[0],'cells':b[1],'rho':b[2],'max_abs_library_correlation':abs(b[2]),'audited_factors':len(rows)}))
