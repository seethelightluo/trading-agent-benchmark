"""Fast fixed-spec validation (one idea): inverse EURUSD beta during commodity/equity divergence."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2033-05-25')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Fixed candidate: inverse EURUSD beta when commodity and equity 5-day returns diverge.
eur=rd('EURUSD',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
eur=eur/(eur.rolling(60,min_periods=40).std()+1e-12)
cmd=R[['XAU','COPPER','WTI']].mean(axis=1).rolling(5,min_periods=4).sum()
eq=R[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1).rolling(5,min_periods=4).sum()
div=(cmd*eq<0)&((cmd-eq).abs()>(cmd-eq).abs().rolling(60,min_periods=40).median())
F=-res(beta(eur,div,30,10),v,peer,dba,trend)"""
assert old in src; src=src.replace(old,new)
src=src.split('# Complete reconstructed admitted-library screen')[0]
src += '''\nprint("FACTOR inverse_eurusd_commodity_equity_divergence_transmission_residual_30 visible_through",E.date(),"assets",len(A))
ics={}
for h in [1,5,10,20]:
 out=[];ns=[];fw=P.shift(-h)/P-1
 for t in P.index:
  q=pd.concat([F.loc[t].rename("f"),fw.loc[t].rename("r")],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((t,q.f.corr(q.r,method="spearman")));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;print(f"h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={x.std(ddof=1)/np.sqrt(len(x)):.6f}")
for n,m in [("2026_27",(ics[20].index>="2026-01-01")&(ics[20].index<"2028-01-01")),("2028_current",ics[20].index>="2028-01-01")]:
 x=ics[20][m];print("regime20",n,"dates",len(x),"IC",f"{x.mean():.6f}","ICIR",f"{x.mean()/x.std(ddof=1):.6f}","hit",f"{(x>0).mean():.4f}")
r=F.rank(axis=1,pct=True);tos=[]
for j in range(1,len(r)):
 q=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(q)>=8:tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method="spearman"))
print("coverage",f"{F.notna().mean().mean():.6f}","valid_cells",int(F.notna().sum().sum()),"turnover",f"{np.mean(tos):.6f}")\n'''
exec(compile(src,'m2_fast','exec'))
