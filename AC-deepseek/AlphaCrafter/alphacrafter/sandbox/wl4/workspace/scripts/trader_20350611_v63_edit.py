"""v63 edit: update docstring + DEFENSIVE_MULT after 2035-05-28..06-11 block."""
import re
from pathlib import Path

p = Path("strategy.py")
src = p.read_text()

# 1) Prepend v63 docstring block after the opening triple-quote start
v63_block = '''"""v63 (2035-06-11):
v62 + three risk-adjustment fires from the 2035-05-28..06-11 block (proposal
on 05-28 EXECUTED, cost = 05-25 closes; account 1223573.54 -> 1216697.17,
-0.56% block, Sharpe -3.34 DD 1.09%; SCREENER ensemble unchanged 0.60/0.40 =
vol_adj_mom_accel_20x60 dir=+1 PRIMARY w=0.60, dn_mkt_beta_60d dir=+1 w=0.40,
loaded live from factors/factor_ensemble.json; root+factors synced
byte-identical; first live block after 4 safety-advance cycles 03-05..05-28):
  - 000688.SH x0.60 -> x0.75  4th consecutive positive (+6.22%/+5.52%/
    +0.99%/+0.43%); fires the v60 watch "000688 x0.75 on 1 more pos".
  - NDX x0.60 -> x1.00  RE-BOOST 2 consecutive positive (+5.36%/+2.01%);
    fires the v60 watch "NDX re-boost x1.00 on 2 cons pos".
  - COPPER x0.70 -> x0.55  3rd consecutive negative (-3.53%/-4.68%/-3.56%);
    fires the v60 watch "COPPER x0.55 on 3rd cons neg or <-8%".
  Kept: SPX x1.00 (2nd cons pos +5.89%/+2.01%; x0.85 on 2nd cons neg or
  <-8%), XAU x1.00 (1st neg -3.39% after 4-pos run; x0.85 on 2nd cons neg or
  <-5%), SX5E x0.70 (1st neg -5.79% LARGE single after 2-pos run; x0.50 on
  2nd cons neg or <-8%), SOX x0.25 (1st pos +8.65% after 2 negs; re-boost
  x0.35 on 2 cons pos), WTI x0.15 (1st pos +9.12% after -11.38% cut;
  re-boost x0.20 on 2 cons pos), N225 x0.50 (1st neg -1.26% after 3-pos run;
  x0.60 on 1 more pos, x0.25 on 4th cons neg or <-6%), US10Y x0.30 (2nd cons
  neg -0.46%/-4.35%; x0.20 on 3rd cons neg, x0.35 on 2 cons pos).
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~39% book
  neutral rank, pnl 0; BTC/ETH/CN10Y multipliers held x0.15/x0.75/x0.70).
  Block: defensive re-tilt HURT - XAU -3.39% (w~0.121 cap) and US10Y -4.35%
  (w~0.038) both negative while momentum names led (WTI +9.12% w~0.026,
  SOX +8.65% w~0.022, SPX +2.01% w~0.099, NDX +2.01% w~0.081); SX5E -5.79%
  (w~0.053 MAIN DRAG ~-0.31%) and COPPER -3.56% (w~0.114 ~-0.41%) dragged.
  Ensemble 0.60/0.40 defensive floor (dn_mkt_beta 0.40) underperformed this
  block - momentum names were the winners. FEEDBACK TO SCREENER: defensive
  re-tilt not validated in current regime; consider trimming dn_mkt_beta
  weight back toward 0.30-0.35 (momentum leaders SPX/NDX/SOX/WTI resilient).
  VIX watch unchanged: break >14 => defensive re-tilt.
'''

# remove old first docstring block (starts with the v62 docstring right after """")
# Find the first occurrence of '"""v62' and the matching closing '"""'
start = src.index('"""v62')
end = src.index('"""', start + 3) + 3
# There is a second docstring """v56...""" later - only replace the first block
src = src[:start] + v63_block + src[end:]

# 2) Update DEFENSIVE_MULT entries
src = src.replace('"NDX": 0.60, "ETH": 0.75, "WTI": 0.15, "BTC": 0.15, "N225": 0.50,',
                  '"NDX": 1.00, "ETH": 0.75, "WTI": 0.15, "BTC": 0.15, "N225": 0.50,')
src = src.replace('"SX5E": 0.70, "SPX": 1.00, "000688.SH": 0.60, "COPPER": 0.70,',
                  '"SX5E": 0.70, "SPX": 1.00, "000688.SH": 0.75, "COPPER": 0.55,')

# 3) Update the v60 comment annotation on the NDX / 000688 / COPPER lines
src = src.replace(
    'NDX kept x0.60 1st pos +5.36% after -6.58% cut, re-boost x1.00 on 2 cons pos, x0.45 on 2nd cons neg or <-6%;',
    'NDX x0.60->x1.00 RE-BOOST 2 cons pos +5.36%/+2.01%, x0.45 on 2nd cons neg or <-6%;')
src = src.replace(
    '000688 x0.45->x0.60 3rd cons pos +6.22%/+5.52%/+0.99%, x0.75 on 1 more pos, x0.25 on 4th cons neg or <-8%;',
    '000688 x0.60->x0.75 4th cons pos +6.22%/+5.52%/+0.99%/+0.43%, x0.25 on 4th cons neg or <-8%;')
src = src.replace(
    'COPPER x0.85->x0.70 2nd cons neg -3.53%/-4.68%, x0.55 on 3rd cons neg or <-8%, re-boost x0.85 on 2 cons pos',
    'COPPER x0.70->x0.55 3rd cons neg -3.53%/-4.68%/-3.56%, x0.40 on 4th cons neg or <-8%, re-boost x0.85 on 2 cons pos')

p.write_text(src)
print("v63 edit done")
