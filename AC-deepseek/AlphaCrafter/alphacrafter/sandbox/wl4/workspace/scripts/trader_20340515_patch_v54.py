import io

p = "strategy.py"
src = open(p, encoding="utf-8").read()

v54 = '''"""v54 (2034-05-15):
v53 + SCREENER 2034-05-01 ensemble re-tilt loaded live (0.60/0.40 two factors:
vol_adj_mom_accel_20x60 dir=+1 w=0.60 PRIMARY, dn_mkt_beta_60d dir=+1 w=0.40;
rate_beta_cn10y_60d EXCLUDED - degenerate flat CN10Y, all betas NaN; root
factors/factor_ensemble.json is the only persisted artifact this cycle) plus
risk-adjustment fires from the 2034-05-01..05-15 block (proposal on 05-01
EXECUTED, gate passed, cost = 04-28 closes; account 1143660.77 -> 1125619.42,
-1.59% block, Sharpe -6.51 DD 1.72%; medium-high VIX 30.6 corrective tape):
  - COPPER cap 0.12 -> x0.85  PRECAUTIONARY TRIM: large single -6.91% at TOP
    live weight w~0.103 (MAIN DRAG ~-0.71%) right after the long cap run;
    mirrors the v44 000688 precautionary-trim precedent (large single at big
    weight after a top run).
  - 000688.SH x0.70 -> x0.45  2nd consecutive negative AND <-8% single
    (-3.00% 03-06..03-20, then -8.74% 05-01..05-15); fires the v53 watch
    "000688 x0.45 on 2nd cons neg or <-8%".
  - N225 x0.50 -> x0.40  2nd consecutive negative (-6.50%, then -5.97%);
    fires the v53 watch "N225 x0.40 on 2nd cons neg or another large <-6%".
  - WTI x0.20 -> x0.15  LARGE AIR-POCKET -14.32% single (1st after 2 stable
    pos); fires the standing WTI air-pocket pattern (impact tiny at w~0.0155).
  - SX5E x0.70 -> x0.60  2nd consecutive negative (-4.96%, then -0.20% mild);
    fires the v53 watch "SX5E x0.60 on 2nd cons neg or large <-8%".
  - SOX x0.30 -> x0.35  3rd consecutive positive (+9.83%/+9.95%/+8.50%);
    fires the v53 watch "SOX x0.35 on 3rd cons pos".
  Kept: XAU x1.00 (2nd cons pos +3.89%/+1.44% at cap; x0.85 on 2nd cons neg
  or <-5%), SPX x0.75 (pos +1.80% resets neg-count; x0.65 on 2nd cons neg or
  <-8%, re-boost x0.85 on 2 cons pos), NDX x0.60 (pos +1.11% 1st after cut;
  re-boost x0.75 on 2 cons pos, x0.45 on 2nd cons neg or <-6%), US10Y x0.30
  (neg -1.35% 1st after pos; x0.25 on 2nd cons neg, re-boost x0.35 on 2 cons
  pos), CN10Y x0.70/ETH x0.75/BTC x0.15 frozen stale.
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~36% book
  neutral rank, pnl 0).
  Ensemble: 0.60/0.40 (SCREENER 2034-05-01 re-tilt) - loaded live from
  factors/factor_ensemble.json.
"""

if not src.startswith('"""v54'):
    src = v54 + "\n\n" + src

o1 = '"XAU": 1.00, "US10Y": 0.30, "CN10Y": 0.70,'
n1 = '"XAU": 1.00, "US10Y": 0.30, "CN10Y": 0.70, "COPPER": 0.85,'
assert o1 in src, "o1 not found"
src = src.replace(o1, n1, 1)

o2 = '"SOX": 0.30, "NDX": 0.60, "ETH": 0.75, "WTI": 0.20, "BTC": 0.15, "N225": 0.50,'
n2 = '"SOX": 0.35, "NDX": 0.60, "ETH": 0.75, "WTI": 0.15, "BTC": 0.15, "N225": 0.40,'
assert o2 in src, "o2 not found"
src = src.replace(o2, n2, 1)

o3 = '"SX5E": 0.70, "SPX": 0.75, "000688.SH": 0.70,'
n3 = '"SX5E": 0.60, "SPX": 0.75, "000688.SH": 0.45,'
assert o3 in src, "o3 not found"
src = src.replace(o3, n3, 1)

open(p, "w", encoding="utf-8").write(src)
print("patched OK")
