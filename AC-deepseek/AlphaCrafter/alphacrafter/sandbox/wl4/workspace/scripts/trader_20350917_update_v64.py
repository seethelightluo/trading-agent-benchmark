from pathlib import Path

p = Path("strategy.py")
s = p.read_text()

v64 = '''"""v64 (2035-09-17):
v63 + one risk-adjustment fire from the 2035-09-03..09-17 block (proposal on
09-03 EXECUTED, cost = 08-31 closes; account 1271004.12 -> 1281954.32,
+0.86% block, Sharpe 3.16 DD 1.08%; SCREENER ensemble unchanged 0.60/0.40 =
vol_adj_mom_accel_20x60 dir=+1 PRIMARY w=0.60, dn_mkt_beta_60d dir=+1 w=0.40,
loaded live from factors/factor_ensemble.json; root+factors synced
byte-identical; first live block after 2035-07-09/07-23 safety advances):
  - WTI x0.15 -> x0.20  RE-BOOST 2 consecutive positive (+9.12% 05-28..06-11,
    +6.30% 09-03..09-17); fires the v63 watch "WTI re-boost x0.20 on 2 cons
    pos".
  Kept: XAU x1.00 (2nd cons pos +6.06% at cap; x0.85 on 2nd cons neg or
  <-5%), NDX x1.00 (2nd cons pos +11.74% TOP; x0.45 on 2nd cons neg or
  <-6%), SPX x1.00 (1st neg -0.84% after 2-pos run; x0.85 on 2nd cons neg or
  <-8%), SX5E x0.70 (1st neg -3.60% after 2-pos run; x0.50 on 2nd cons neg or
  <-8%), 000688.SH x0.75 (1st neg -1.10% after 4-pos run; x0.60 on 2nd cons
  neg or <-8%), SOX x0.25 (1st neg -7.42% after +8.65%; x0.20 on 3rd cons neg
  or <-10%), N225 x0.50 (2nd cons neg -1.26%/-3.67%; x0.25 on 4th cons neg or
  <-6%), COPPER x0.55 (1st pos +2.99% after 3 negs; x0.40 on 4th cons neg or
  <-8%, re-boost x0.85 on 2 cons pos), US10Y x0.30 (1st pos +0.83% after 2
  negs; x0.35 on 2 cons pos, x0.20 on 3rd cons neg).
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~39% book
  neutral rank, pnl 0).
  Block: momentum/commodity led - NDX +11.74% (w~0.06), XAU +6.06% (w~0.12
  cap TOP contributor), WTI +6.30% (w~0.016), COPPER +2.99% (w~0.056);
  tech/semis & Europe dragged - SOX -7.42% (w~0.025), SX5E -3.60% (w~0.066
  MAIN DRAG ~-0.24%), N225 -3.67% (w~0.043), SPX -0.84% (w~0.118), 000688
  -1.10% (w~0.066). Ensemble 0.60/0.40 momentum-primary validated (NDX/XAU/
  WTI/COPPER winners captured; SOX/N225/SX5E contained by defensive mults).
  VIX watch unchanged: break >14 => defensive re-tilt.
"""

anchor = '"""v63 (2035-06-11):'
assert anchor in s, "v63 docstring anchor not found"
s = s.replace(anchor, v64 + anchor, 1)

old = '"SOX": 0.25, "NDX": 1.00, "ETH": 0.75, "WTI": 0.15, "BTC": 0.15, "N225": 0.50,'
new = '"SOX": 0.25, "NDX": 1.00, "ETH": 0.75, "WTI": 0.20, "BTC": 0.15, "N225": 0.50,'
assert old in s, "WTI dict line not found"
s = s.replace(old, new, 1)

p.write_text(s)
print("strategy.py updated to v64")
