# append 40d ICIR diagnostic
p='scripts/miner_1_20320108_macro_stress_reversal.py'
s=open(p).read(); s=s.replace("for h in [1,5,10,20,40]: print('decay',h,calc(h).mean())", "for h in [1,5,10,20,40]:\n z=calc(h); print('decay',h,z.mean(),z.mean()/z.std(ddof=1),len(z))")
open(p,'w').write(s)
