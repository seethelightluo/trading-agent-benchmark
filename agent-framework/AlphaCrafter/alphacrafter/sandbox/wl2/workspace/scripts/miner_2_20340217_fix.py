# Change validated signal to reversal orientation, then recompute artifacts.
p='scripts/miner_2_20340217_risk_adjusted_trend.py'
s=open(p).read(); s=s.replace("f=(mom/vol).shift(1).rank(axis=1,pct=True)","f=(-(mom/vol)).shift(1).rank(axis=1,pct=True)")
open(p,'w').write(s)
