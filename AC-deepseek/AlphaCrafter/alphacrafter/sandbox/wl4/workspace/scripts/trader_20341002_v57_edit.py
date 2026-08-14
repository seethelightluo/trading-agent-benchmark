# -*- coding: utf-8 -*-
"""Trader: update strategy.py from v56 -> v57 (risk-adjustment fires from the
2034-09-18..10-02 block)."""
import io
import re

p = 'strategy.py'
src = io.open(p, encoding='utf-8').read()

v57 = '''"""v57 (2034-10-02):
v56 + seven risk-adjustment fires from the 2034-09-18..10-02 block (proposal
on 09-18 PERSISTED/SKIPPED by deterministic gate - holdings unchanged at the
09-04 target; account 1163713.34 -> 1165658.00, +0.17% block, Sharpe 0.66 DD
0.79%; SCREENER ensemble unchanged 0.65/0.35 = vol_adj_mom_accel_20x60 dir=+1
PRIMARY w=0.65, dn_mkt_beta_60d dir=+1 w=0.35, loaded live from
factors/factor_ensemble.json; root+factors synced byte-identical):
  - SOX x0.30 -> x0.25  2nd consecutive negative (-4.94%, then -18.85% LARGE
    <-10%); fires the v56 watch "SOX x0.25 on 2nd cons neg or <-10%".
  - COPPER (cap) -> x0.85  2nd consecutive negative (-5.39%, then -8.05% at
    the ~-8% threshold); fires the v56 watch "COPPER x0.85 on 2nd cons neg
    or <-8%". First explicit COPPER multiplier entry.
  - N225 x0.40 -> x0.30  3rd consecutive negative (-6.50%/-1.87%/-0.26%);
    fires the v56 watch "N225 x0.30 on 3rd cons neg".
  - 000688.SH x0.45 -> x0.30  3rd consecutive negative (-3.00%/-0.37%/
    -1.07%); fires the v56 watch "000688 x0.30 on 3rd cons neg".
  - US10Y x0.30 -> x0.25  2nd consecutive negative (-4.59%/-5.45%); fires
    the v56 watch "US10Y x0.25 on 2nd cons neg".
  - WTI x0.20 -> x0.25  RE-BOOST 2 consecutive positive (+5.76%/+20.16%);
    fires the v56 watch "WTI x0.25 on 1 more cons pos".
  - NDX x0.60 -> x1.00  RE-BOOST 2 consecutive positive (+5.39%/+4.60%);
    fires the v56 watch "NDX x1.00 on 2 cons pos".
  - SPX x0.75 -> x0.85  RE-BOOST 2 consecutive positive (+4.67%/+7.26%);
    fires the v56 watch "SPX x0.85 on 2 cons pos".
  Kept: XAU x1.00 (2nd cons pos +5.93%; x0.85 on 2nd cons neg or <-5%),
  SX5E x0.60 (1st pos +3.08% after 2 negs; re-boost x0.70 on 2 cons pos,
  x0.50 on 2nd cons neg).
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~33% book
  neutral rank, pnl 0).
  Block: risk-on rotation continued - WTI +20.16% / SPX +7.26% / XAU +5.93%
  / NDX +4.60% / SX5E +3.08% led; SOX -18.85% (MAIN DRAG, contained by
  x0.30 mult) and COPPER -8.05% dragged; US10Y -5.45% on rate creep. The
  screener's mean-reversion flag fully materialized (SOX/COPPER momentum
  leaders reversing hard, defensive/quality leading).
  Ensemble: 0.65/0.35 (SCREENER 2034-09-04 re-tilt) - loaded live.
"""
'''

anchor = '"""v55 (2034-09-04):'
assert anchor in src, 'anchor v55 docstring not found'
src = src.replace(anchor, v57 + '\n' + anchor, 1)

new_mult = '''DEFENSIVE_MULT = {
    "XAU": 1.00, "US10Y": 0.25, "CN10Y": 0.70,   # safe havens (v57: XAU kept x1.00 2nd cons pos +5.93%, x0.85 on 2nd cons neg or <-5%; US10Y x0.30->x0.25 2nd cons neg -4.59%/-5.45%, x0.20 on 3rd cons neg, re-boost x0.30 on 2 cons pos; CN10Y kept x0.70 frozen stale)
    "SOX": 0.25, "NDX": 1.00, "ETH": 0.75, "WTI": 0.25, "BTC": 0.15, "N225": 0.30,  # high-beta (v57: SOX x0.30->x0.25 2nd cons neg -4.94%/-18.85% LARGE <-10%, x0.20 on 3rd cons neg or <-10%, re-boost x0.35 on 2 cons pos; NDX x0.60->x1.00 2 cons pos +5.39%/+4.60%, x0.60 on 2nd cons neg or <-6%; WTI x0.20->x0.25 2 cons pos +5.76%/+20.16%, x0.30 on 1 more pos, x0.15 on 2nd cons neg or <-8%; N225 x0.40->x0.30 3rd cons neg -6.50%/-1.87%/-0.26%, x0.25 on 4th cons neg or <-6%, re-boost x0.40 on 2 cons pos; BTC/ETH frozen stale)
    "SX5E": 0.60, "SPX": 0.85, "000688.SH": 0.30, "COPPER": 0.85,  # v57: SX5E kept x0.60 1st pos +3.08% after 2 negs, re-boost x0.70 on 2 cons pos, x0.50 on 2nd cons neg; SPX x0.75->x0.85 2 cons pos +4.67%/+7.26%, x1.00 on 3rd cons pos, x0.65 on 2nd cons neg or <-8%; 000688 x0.45->x0.30 3rd cons neg -3.00%/-0.37%/-1.07%, x0.25 on 4th cons neg or <-8%, re-boost x0.45 on 2 cons pos; COPPER cap->x0.85 2nd cons neg -5.39%/-8.05% (~-8% thresh), x0.70 on 3rd cons neg or <-8%, re-boost cap on 2 cons pos
}'''

pat = re.compile(r'DEFENSIVE_MULT = \{.*?\n\}', re.S)
src2, n = pat.subn(new_mult, src, count=1)
assert n == 1, 'DEFENSIVE_MULT not replaced'
io.open(p, 'w', encoding='utf-8').write(src2)
print('OK: v57 docstring inserted, DEFENSIVE_MULT replaced')
