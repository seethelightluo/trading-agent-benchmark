p='scripts/miner_1_20320527_inverse_residual_drawdown_dispersion_conditioned_recovery_20_60d.py'
s=open(p).read()
s=s.replace("s=np.std(v[:-1],ddof=1); shock=v[:-1] < -s\n  return np.mean(v[1:][shock])/s if np.isfinite(s) and s>1e-12 and shock.any() else np.nan", "v=np.asarray(v,float); good=np.isfinite(v); s=np.std(v[:-1][np.isfinite(v[:-1])],ddof=1); shock=(v[:-1] < -s) & np.isfinite(v[1:])\n  return np.mean(v[1:][shock])/s if np.isfinite(s) and s>1e-12 and shock.any() else np.nan")
open(p,'w').write(s)
