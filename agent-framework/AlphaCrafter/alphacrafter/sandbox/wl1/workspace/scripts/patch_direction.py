# Patch artifact direction to match persisted contrarian weakness definition.
p='scripts/miner_3_20330107_downside_resilient_lead.py'
s=open(p).read(); s=s.replace("out=f.stack().rename('signal').reset_index()", "out=(-f).stack().rename('signal').reset_index()")
open(p,'w').write(s)
