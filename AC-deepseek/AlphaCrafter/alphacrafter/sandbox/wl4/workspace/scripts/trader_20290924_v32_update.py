"""Apply v32 header + DEFENSIVE_MULT update to strategy.py (fired 2029-09-24)."""
import re
from pathlib import Path

p = Path("strategy.py")
src = p.read_text(encoding="utf-8")

V32_HEADER = '''Trader strategy v32 on top of v31 (fired 2029-09-24 after the 09-10..09-24
block -1.94%, DD 2.75%, Sharpe -5.63 - NEGATIVE block, commodity/rate reversal:
09-10 proposal EXECUTED (gate passed; v31 target migrated at 09-10 closes;
account 919457.17->901590.02 (-1.94%; ~-9.85% since 07-16 online start, cash 0,
15 positions, weights-sum 1.0, no open orders); block winners SOX +2.61%
(w~5.4%, +0.14%) SPX +1.10% (w~5.7%, +0.06%) COPPER +0.35% (w~7.6%) NDX +0.34%
(w~6.4%); losers WTI -13.71% (w~7.0%, MAIN DRAG ~-0.95%) US10Y -2.33% (w~12.0%,
~-0.28%) SX5E -2.57% (w~10.7%, ~-0.28%) CN10Y -6.45% (w~3.6%, ~-0.23%) 000688
-2.14% (w~6.8%, ~-0.15%) N225 -1.76% BTC -6.32% (w~0.6%, ~-0.04% contained by
x0.15) XAU -0.15%; broad commodity/rate reversal (WTI +8.87% then -13.71%,
CN10Y +8.40% then -6.45%, US10Y +1.29% then -2.33%) offset by tech stabilization
(SOX/SPX/NDX positive after v31 cuts); v31 defensive cuts WORKED (SPX/NDX/SOX/
BTC all positive or contained); stale-guard covers 000300.SH/HSI/ETH frozen
(pnl 0, ~23% book neutral rank); no execution/order anomalies.
v32 changes (evidence from the 09-10..09-24 block; v30/v31 watches fired):
  1. WTI x1.00 -> x0.80 (LARGE AIR-POCKET FIRED: -13.71% immediately after the
     +8.87% top-winner block; v30 watch said cut on large <-10%; v24 precedent
     (large print after re-boost payoff -> revert to pre-re-boost depth x0.80);
     MAIN DRAG ~-0.95%).
  2. CN10Y x0.80 -> x0.70 (LARGE AIR-POCKET AFTER RE-BOOST FIRED: -6.45% after
     +8.40%; mirror of the v24 WTI precedent - revert to pre-re-boost depth
     x0.70; rate_beta_cn10y_60d dir=-1 PRIMARY keeps the rate-hedge short).
  3. SX5E x0.85 -> x0.70 (WATCH-CUT FIRED: 2nd consecutive negative -3.85% /
     -2.57%; v31 watch said cut on 2nd cons neg; 2nd-largest book weight 10.7%).
  4. 000688.SH x0.85 -> x0.70 (WATCH-CUT FIRED: 2nd consecutive negative
     -11.28% / -2.14%; v31 watch said cut on 2nd cons neg).
Kept: SPX x0.55 (+1.10% 1st pos after 3 neg; re-boost x0.65 on 2 cons pos),
NDX x0.55 (+0.34%), SOX x0.45 (+2.61%), BTC x0.15 (contained -6.32% at 0.6%
wt; deepen x0.10 on another large print or 2 more cons neg), XAU x0.85
(-0.15% flat), US10Y x1.00 (single -2.33%; de-boost x0.85 on 2nd cons neg),
N225 x0.85 (mild -1.76%), COPPER x1.00 (+0.35%), ETH x0.75 (frozen),
CAP 0.12, SPREAD 0.06, vol exp 0.6, stale-guard, 10-day cadence.
New watches: WTI (x0.80->x0.65 on 2nd cons large or another <-10%), CN10Y
(x0.70->x0.60 on 2nd cons neg or another large <-6%), SX5E (x0.70->x0.60 on
3rd cons neg or large <-8%), 000688 (x0.70->x0.60 on 3rd cons neg or large
<-8%), US10Y (x0.85 on 2nd cons neg), SPX (x0.65 on 2 cons pos), NDX (x0.65
on 2 cons pos), SOX (x0.55 on 2 cons pos), BTC (x0.10 on another large print
or 2 more cons neg), XAU (x1.00 on 2 cons pos), N225 (x0.70 on 3rd cons large
or <-6%), COPPER (cut on 2nd cons large).

'''

NEW_MULT = '''DEFENSIVE_MULT = {
    "XAU": 0.85, "US10Y": 1.00, "CN10Y": 0.70,   # safe havens (v32: CN10Y x0.80->x0.70 LARGE AIR-POCKET AFTER RE-BOOST -6.45% after +8.40%, v24 WTI precedent revert to pre-re-boost depth; XAU kept x0.85 -0.15% flat, re-boost x1.00 on 2 cons pos; US10Y kept x1.00 single -2.33%, de-boost x0.85 on 2nd cons neg)
    "SOX": 0.45, "NDX": 0.55, "ETH": 0.75, "WTI": 0.80, "BTC": 0.15, "N225": 0.85,  # high-beta cuts (v32: WTI x1.00->x0.80 LARGE AIR-POCKET -13.71% after +8.87% top-winner block, v24 precedent revert to pre-re-boost depth; SOX kept x0.45 +2.61% pos, re-boost x0.55 on 2 cons pos; NDX kept x0.55 +0.34% 1st pos after 3 neg; BTC kept x0.15 -6.32% contained at 0.6% wt; N225 kept x0.85 mild -1.76%)
    "SX5E": 0.70, "SPX": 0.55, "000688.SH": 0.70,  # v32: SX5E x0.85->x0.70 WATCH-CUT 2nd cons neg -3.85%/-2.57%; SPX kept x0.55 +1.10% 1st pos after 3 neg (re-boost x0.65 on 2 cons pos); 000688 x0.85->x0.70 WATCH-CUT 2nd cons neg -11.28%/-2.14%
}'''

# 1) Insert v32 header at top of docstring
assert src.startswith('"""'), "expected docstring start"
doc_end = src.index('"""', 3)
body = src[doc_end + 3:]
new_src = '"""' + V32_HEADER + src[3:doc_end] + '"""' + body

# 2) Replace DEFENSIVE_MULT dict
pat = re.compile(r"DEFENSIVE_MULT = \{.*?\n\}", re.S)
new_src2, n = pat.subn(NEW_MULT, new_src, count=1)
assert n == 1, "DEFENSIVE_MULT replacement failed"

p.write_text(new_src2, encoding="utf-8")
print("v32 header inserted, DEFENSIVE_MULT replaced, bytes:", len(new_src2))
