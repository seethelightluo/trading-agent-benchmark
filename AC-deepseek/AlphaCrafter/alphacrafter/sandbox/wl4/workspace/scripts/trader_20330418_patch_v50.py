"""Trader 2033-04-18: patch strategy.py to v50 (six risk-adjustment fires + sync ensemble)."""
import shutil, hashlib

# sync root ensemble to factors copy (loader reads factors/ live)
shutil.copyfile('factors/factor_ensemble.json', 'factor_ensemble.json')
print('root  ', hashlib.md5(open('factor_ensemble.json','rb').read()).hexdigest())
print('factors', hashlib.md5(open('factors/factor_ensemble.json','rb').read()).hexdigest())

src = open('strategy.py').read()

old_mult_start = src.index('DEFENSIVE_MULT = {')
old_mult_end = src.index('}', src.index('DEFENSIVE_MULT = {')) + 1
old_mult = src[old_mult_start:old_mult_end]

new_mult = '''DEFENSIVE_MULT = {
    "XAU": 1.00, "US10Y": 0.30, "CN10Y": 0.70,   # safe havens (v50: XAU kept x1.00 1st pos +0.83% after neg, x0.85 on 2nd cons neg or <-5%; US10Y kept x0.30 1st pos +2.79%, re-boost x0.35 on 2 cons pos; CN10Y kept x0.70 frozen stale)
    "SOX": 0.30, "NDX": 0.75, "ETH": 0.75, "WTI": 0.15, "BTC": 0.15, "N225": 0.50,  # high-beta (v50: N225 x0.60->x0.50 ANOTHER large -6.47% after -7.65% chain; WTI x0.20->x0.15 2nd cons neg -9.17%/-6.08%; SOX x0.20->x0.30 RE-BOOST 2 cons pos +9.36%/+16.32%; NDX kept x0.75 1st neg -3.85%, re-boost x1.00 on 2 cons pos; BTC/ETH frozen stale)
    "SX5E": 0.60, "SPX": 0.65, "000688.SH": 0.45,  # v50: SX5E x0.50->x0.60 RE-BOOST 2 cons pos +5.86%/+2.22%; SPX x0.75->x0.65 2nd cons neg -5.93%/-1.14%; 000688 x0.60->x0.45 2nd cons neg -6.93%/-1.68%
}'''
src = src.replace(old_mult, new_mult)

v50_header = '''v50 (2033-04-18):
v49 + six risk-adjustment fires from the 04-05..04-18 block (window was
advanced/safety-drifted with the 03-21 target; account 1118545.22 ->
1123336.25, +0.43% block, holdings unchanged at 03-21 target, cost = 03-18
closes; block rets 04-04->04-15): SOX +16.32% TOP, COPPER +4.49%, US10Y
+2.79%, SX5E +2.22%, XAU +0.83%, SPX -1.14%, 000688.SH -1.68%, NDX -3.85%,
WTI -6.08%, N225 -6.47%; frozen stale names pnl 0 (000300.SH/HSI/BTC/ETH/
CN10Y ~37% book neutral rank):
  - N225 x0.60 -> x0.50  ANOTHER large single negative -6.47% on top of the
    -7.65% cut block (chain: x1.00->x0.85->x0.70->x0.60->x0.50); fires the
    v49 watch "N225 x0.50 on another large <-6% or 2nd cons neg"; MAIN DRAG
    ~-0.41% at w~0.063.
  - WTI x0.20 -> x0.15  2nd consecutive negative block (-9.17% then -6.08%);
    fires the v49 watch "WTI x0.15 on another large or 2nd cons neg";
    repeated energy weakness, w~0.014 tiny impact.
  - SOX x0.20 -> x0.30  RE-BOOST 2 consecutive STRONG positive blocks
    (+9.36% then +16.32% TOP winner); fires the v49 watch "SOX re-boost
    x0.30 on 2 cons pos".
  - SX5E x0.50 -> x0.60  RE-BOOST 2 consecutive positive blocks (+5.86% then
    +2.22%); fires the v49 watch "SX5E re-boost x0.60 on 2 cons pos".
  - SPX x0.75 -> x0.65  2nd consecutive negative block (-5.93% then -1.14%);
    fires the v49 watch "SPX x0.65 on 2nd cons neg or <-8%".
  - 000688.SH x0.60 -> x0.45  2nd consecutive negative block (-6.93% then
    -1.68%); fires the v49 watch "000688 x0.45 on 2nd cons neg or <-8%".
  Kept: XAU x1.00 (1st pos +0.83% after the 1st-neg; x0.85 on 2nd cons neg or
  <-5%), US10Y x0.30 (1st pos +2.79% after mild neg; re-boost x0.35 on 2 cons
  pos), NDX x0.75 (single neg -3.85% after the +12.24% top; re-boost x1.00 on
  2 cons pos), COPPER cap 0.12 (pos +4.49%), CN10Y x0.70/ETH x0.75/BTC x0.15
  frozen stale.
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~37% book
  neutral rank).
  Ensemble: SCREENER RE-TILT 0.50/0.25/0.25 (vol_adj_mom_accel_20x60 dir=+1
  PRIMARY w=0.50, rate_beta_cn10y_60d dir=-1 w=0.25, dn_mkt_beta_60d dir=+1
  w=0.25; +0.10 primary, -0.10 dn_beta vs 05-31 tilts) - loaded live from
  factors/factor_ensemble.json (root synced byte-identical).

'''
assert src.startswith('"""')
src = '"""' + v50_header + src[len('"""'):]
open('strategy.py', 'w').write(src)
print('patched OK -> v50')
