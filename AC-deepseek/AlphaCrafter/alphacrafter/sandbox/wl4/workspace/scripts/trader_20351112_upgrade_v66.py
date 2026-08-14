"""Upgrade strategy.py to v66: SOX x0.15 -> x0.10 (5th consecutive negative)."""
p = "strategy.py"
s = open(p, encoding="utf-8").read()

v66 = '''"""v66 (2035-11-12):
v65 + one risk-adjustment fire from the 2035-10-29..11-09 block (proposal on
10-29 EXECUTED, cost = 10-26 closes; account 1269665.31 -> 1291586.71,
+1.73% block, gross/net 100%, cash 0, 15 positions, no open orders; SCREENER
ensemble unchanged 0.60/0.40 = vol_adj_mom_accel_20x60 dir=+1 PRIMARY w=0.60,
dn_mkt_beta_60d dir=+1 w=0.40, loaded live from factors/factor_ensemble.json;
root+factors synced byte-identical md5 ace14c70):
  - SOX x0.15 -> x0.10  5th consecutive negative (-7.42%/-0.79%/-15.55%/
    -1.69%/-2.04%); fires the v65 watch "SOX x0.10 on 5th cons neg or
    <-15%". Floor reached at x0.10; x0.05 on 6th cons neg or <-15%,
    re-boost x0.20 on 2 cons pos.
  Kept: XAU x0.85 (1st pos +6.67% after -5.75% cut; x1.00 on 2 cons pos,
  x0.70 on 2nd cons neg or <-8%), SPX x0.85 (1st pos +2.20% after 2 negs;
  x1.00 on 2 cons pos, x0.70 on 3rd cons neg or <-8%), N225 x0.60 (1st pos
  +3.64% after -1.53%; x0.70 on 2 cons pos, x0.25 on 4th cons neg or <-6%),
  SX5E x0.80 (3rd cons pos +1.95%; cap 0.12 effective, x0.50 on 2nd cons neg
  or <-8%), COPPER x0.85 (3rd cons pos +3.14%; x0.40 on 4th cons neg or
  <-8%), NDX x1.00 (1st pos +1.36% after -2.62%; x0.45 on 2nd cons neg or
  <-6%), WTI x0.20 (1st mild neg -0.17% after 4-pos run; x0.10 on 2nd cons
  neg or <-8%, x0.25 on 5 cons pos), US10Y x0.30 (1st pos +3.14% after
  -1.66%; x0.35 on 2 cons pos, x0.20 on 3rd cons neg), 000688.SH x0.15 (1st
  pos +0.77% after 4 negs; x0.10 on 5th cons neg or <-12%, re-boost x0.30 on
  2 cons pos).
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~39% book
  neutral rank, pnl 0).
  Block: broad risk-on - XAU +6.67% (w~0.085 TOP) N225 +3.64% (w~0.097) SPX
  +2.20% COPPER +3.14% US10Y +3.14% SX5E +1.95% NDX +1.36% led; SOX -2.04%
  (w~0.018, contained by x0.15) and WTI -0.17% lagged. VIX 12.21 (+14.6%
  over 10d, below the >14 defensive-re-tilt watch); XAU rebound validates
  keeping the safe-haven floor; SOX now 5 cons neg at x0.10 floor - tech
  breadth remains weak. FEEDBACK TO SCREENER: momentum up-tilt (0.60) +
  downside-beta overlay (0.40) both paid (XAU/N225/COPPER winners, SOX
  underweight); consider whether SOX/NDX tech-weakness vs commodity/energy
  strength rotation persists; VIX still below 14 so no defensive re-tilt.
"""
'''
assert s.startswith('"""v65'), s[:20]
s = v66 + s

old_sox = '"SOX": 0.15,'
new_sox = '"SOX": 0.10,'
assert old_sox in s
s = s.replace(old_sox, new_sox, 1)

old_c = ("SOX x0.20->x0.15 4th cons neg -7.42%/-0.79%/-15.55%/-1.69%, "
         "x0.10 on 5th cons neg or <-15%, re-boost x0.25 on 2 cons pos;")
new_c = ("SOX x0.15->x0.10 5th cons neg -7.42%/-0.79%/-15.55%/-1.69%/-2.04% "
         "(v66), x0.05 on 6th cons neg or <-15%, re-boost x0.20 on 2 cons pos;")
assert old_c in s, "sox comment not found"
s = s.replace(old_c, new_c, 1)

open(p, "w", encoding="utf-8").write(s)
print("strategy.py updated to v66")
