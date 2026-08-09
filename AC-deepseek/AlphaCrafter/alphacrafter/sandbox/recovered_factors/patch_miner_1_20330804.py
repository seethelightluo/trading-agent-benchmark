# Replace brute-force artifact scan in this research script with admitted-factor signal scan.
p='scripts/miner_1_20330804_residual_drawdown_compression_recovery_slope_20d.py'
s=open(p).read()
a=s.index('# Compare to every available factor signal artifact')
b=s.index("f.to_pickle(",a)
new='''# Library overlap: current admitted (non-backup/non-evicted) factor records only.
# Each admitted definition must have a matching persisted signal artifact for admission evidence.
records=[]
for jf in glob.glob('factors/*.json'):
 try:
  rec=pd.read_json(jf,typ='series')
  if str(rec.get('validation',{}).get('status','')).upper()=='EFFECTIVE': records.append(str(rec.get('factor_id')))
 except Exception: pass
maxrho=-1.; maxname=None; compared=0; errors=[]
for fid in records:
 matches=glob.glob('scripts/*'+fid+'*_signal.pkl')
 if not matches:
  errors.append((fid,'missing signal artifact'));continue
 try:
  x=pd.read_pickle(matches[-1]); ds=f.index.intersection(x.index);cs=f.columns.intersection(x.columns); vals=[]
  for d in ds:
   q=pd.concat([f.loc[d,cs].rename('f'),x.loc[d,cs].rename('x')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1 and q.x.nunique()>1:
    z=spearmanr(q.f,q.x).statistic
    if np.isfinite(z):vals.append(abs(z))
  if not vals: raise ValueError('no common IC-sized cross sections')
  compared+=1; z=max(vals)
  if z>maxrho:maxrho,maxname=z,fid
 except Exception as e: errors.append((fid,str(e)))
print('LIBRARY_CORRELATION max_abs=%.6f factor=%s factors=%d expected=%d errors=%d'%(maxrho,maxname,compared,len(records),len(errors)))
if errors: print('LIBRARY_ERRORS',errors[:10])
'''
open(p,'w').write(s[:a]+new+s[b:])
