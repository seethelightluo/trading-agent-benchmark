# Patch revalidation artifact resolution for legacy factor JSON records lacking signal_artifact.
p='scripts/miner_2_20300822_contrarian_range_position_pressure_10x20obs_revalidation.py'
s=open(p).read()
s=s.replace("artifact=d.get('signal_artifact')\n if not artifact or not os.path.exists(artifact): mx=np.inf;ev[fid]={'rho':None,'common_signal_cells':0,'file':artifact};continue", "artifact=d.get('signal_artifact')\n if not artifact or not os.path.exists(artifact):\n  key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')\n  hits=[z for z in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(z)]\n  artifact=max(hits,key=os.path.getmtime) if hits else None\n if not artifact or not os.path.exists(artifact): mx=np.inf;ev[fid]={'rho':None,'common_signal_cells':0,'file':artifact};continue")
open(p,'w').write(s)
