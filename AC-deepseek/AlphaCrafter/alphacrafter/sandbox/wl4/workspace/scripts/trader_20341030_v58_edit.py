from pathlib import Path

p = Path('strategy.py')
s = p.read_text()

v58_doc = '''"""v58 (2034-10-30):
v57 + four risk-adjustment fires from the 2034-10-16..10-30 block (proposal
on 10-16 PERSISTED/SKIPPED by deterministic gate - holdings unchanged at the
09-04 target; account 1148090.23 -> 1140571.08, -0.65% block, Sharpe -2.68
DD 1.23%; SCREENER ensemble unchanged 0.65/0.35 = vol_adj_mom_accel_20x60
dir=+1 PRIMARY w=0.65, dn_mkt_beta_60d dir=+1 w=0.35, loaded live from
factors/factor_ensemble.json; root+factors synced byte-identical):
  - WTI x0.25 -> x0.15  LARGE single -23.30% <-8% air-pocket after the 2-pos
    run (+5.76%/+20.16%); fires the v57 watch "WTI x0.15 on 2nd cons neg or
    <-8%".
  - 000688.SH x0.30 -> x0.25  4th consecutive negative (-3.00%/-0.37%/
    -1.07%/-4.36%); fires the v57 watch "000688 x0.25 on 4th cons neg".
  - COPPER x0.85 -> x0.70  3rd consecutive negative (-5.39%/-8.05%/-4.78%);
    fires the v57 watch "COPPER x0.70 on 3rd cons neg or <-8%".
  - SPX x0.85 -> x1.00  3rd consecutive positive (+4.67%/+7.26%/+1.15%);
    fires the v57 watch "SPX x1.00 on 3rd cons pos".
  Kept: XAU x1.00 (3rd cons pos +1.10%; x0.85 on 2nd cons neg or <-5%),
  US10Y x0.25 (1st non-neg +0.02% after 2 negs; x0.20 on 3rd cons neg,
  re-boost x0.30 on 2 cons pos), SOX x0.25 (1st pos +11.01% after 2 negs;
  re-boost x0.35 on 2 cons pos, x0.20 on 2nd cons neg or <-10%), N225 x0.30
  (1st pos +6.78% after 3 negs; re-boost x0.40 on 2 cons pos, x0.25 on 2nd
  cons neg or <-6%), NDX x1.00 (flat +0.03% 1st after 2 pos; x0.60 on 2nd
  cons neg or <-6%), SX5E x0.60 (1st neg -4.93% after pos; x0.50 on 2nd
  cons neg or <-8%, re-boost x0.70 on 2 cons pos), CN10Y x0.70 frozen.
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~39.5% book
  neutral rank, pnl 0).
  Block: mean-reversion rebound SOX +11.01% / N225 +6.78% vs WTI -23.30%
  (MAIN DRAG ~-0.54% at w~0.023, contained by x0.25 mult) and SX5E -4.93% /
  COPPER -4.78% / 000688 -4.36% drift lower; gate skipped the 10-16
  proposal, holdings static at the 09-04 target the whole block.
  Ensemble: 0.65/0.35 (SCREENER 2034-09-04 re-tilt) - loaded live.
"""

start = s.index('DEFENSIVE_MULT = {')
end = s.index('\n\n\ndef stock')
old_mult = s[start:end]

new_mult = '''DEFENSIVE_MULT = {
    "XAU": 1.00, "US10Y": 0.25, "CN10Y": 0.70,   # safe havens (v58: XAU kept x1.00 3rd cons pos +1.10%, x0.85 on 2nd cons neg or <-5%; US10Y kept x0.25 1st non-neg +0.02% after 2 negs, x0.20 on 3rd cons neg, re-boost x0.30 on 2 cons pos; CN10Y kept x0.70 frozen stale)
    "SOX": 0.25, "NDX": 1.00, "ETH": 0.75, "WTI": 0.15, "BTC": 0.15, "N225": 0.30,  # high-beta (v58: SOX kept x0.25 1st pos +11.01% after 2 negs -18.85%/-4.94%, x0.20 on 2nd cons neg or <-10%, re-boost x0.35 on 2 cons pos; NDX kept x1.00 flat +0.03% after 2 pos, x0.60 on 2nd cons neg or <-6%; WTI x0.25->x0.15 LARGE -23.30% <-8% air-pocket after 2-pos run, x0.15 held on another large or 2nd cons neg, re-boost x0.25 on 2 stable pos; N225 kept x0.30 1st pos +6.78% after 3 negs, x0.25 on 2nd cons neg or <-6%, re-boost x0.40 on 2 cons pos; BTC/ETH frozen stale)
    "SX5E": 0.60, "SPX": 1.00, "000688.SH": 0.25, "COPPER": 0.70,  # v58: SX5E kept x0.60 1st neg -4.93% after pos, x0.50 on 2nd cons neg or <-8%, re-boost x0.70 on 2 cons pos; SPX x0.85->x1.00 3rd cons pos +4.67%/+7.26%/+1.15%, x0.85 on 2nd cons neg or <-8%; 000688 x0.30->x0.25 4th cons neg -4.36%, x0.20 on 5th cons neg or <-8%, re-boost x0.45 on 2 cons pos; COPPER x0.85->x0.70 3rd cons neg -5.39%/-8.05%/-4.78%, x0.60 on 4th cons neg or <-8%, re-boost cap on 2 cons pos
}'''

s = s[:start] + new_mult + s[end:]
s = v58_doc + s
p.write_text(s)
print('OK - v58 written')
print('head:', s[:80].replace(chr(10), ' '))
