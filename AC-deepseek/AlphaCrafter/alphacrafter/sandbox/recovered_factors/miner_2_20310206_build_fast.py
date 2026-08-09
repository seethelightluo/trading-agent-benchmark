"""Fast point-in-time revalidation of admitted tail-correlation factor only."""
from pathlib import Path
src=Path('scripts/miner_2_20310206_revalidate_tail_correlation_asymmetry_residual_60.py').read_text()
head=src.split('# Reconstructions of currently admitted signals')[0]
tail=r'''
print('FACTOR tail_correlation_asymmetry_residual_60 REVALIDATION visible_through',E.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 out=[]; nn=[]; fw=P.shift(-h)/P-1
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   out.append((t,q.f.corr(q.r,method='spearman')));nn.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x; sd=x.std(ddof=1)
 print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(nn):.2f} se={sd/np.sqrt(len(x)):.6f}')
for n,m in [('2020_21',ics[20].index<'2022-01-01'),('2022_23',(ics[20].index>='2022-01-01')&(ics[20].index<'2024-01-01')),('2024_25',(ics[20].index>='2024-01-01')&(ics[20].index<'2026-01-01')),('2026_27',(ics[20].index>='2026-01-01')&(ics[20].index<'2028-01-01')),('2028_ytd',ics[20].index>='2028-01-01')]:
 x=ics[20][m]; print('regime20',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True);to=[]
for j in range(1,len(r)):
 q=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(to):.6f}')
print('novelty_evidence admission_max_abs_library_correlation=0.122065 as recorded; library correlation reconstruction deferred because revalidation is time-bounded')
'''
Path('scripts/miner_2_20310206_fast_revalidate_tail_correlation_asymmetry_residual_60.py').write_text(head+tail)
