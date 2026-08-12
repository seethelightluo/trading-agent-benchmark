"""Trader v37 update: patch strategy.py DEFENSIVE_MULT + prepend v37 doc header."""
with open('strategy.py') as f:
    content = f.read()

v37_doc = '''"""Trader strategy v37 on top of v36 (fired 2030-12-16 after the 12-02..12-16 block
+1.15%, DD 0.28%, Sharpe 6.24, Calmar 111.56 - POSITIVE block; 12-02 proposal
EXECUTED (gate passed, migrated to the v36 target at 12-02 closes; cost prices
= 12-02 closes); account 956121.98->967101.17 (+1.15%; ~-3.3% since 07-16 online
start, cash 0, 15 positions, weights-sum 1.0, no open orders); block winners
COPPER +16.69% (w~7.2%, +1.19% contrib, BIG rebound after the -7.02% single
large - 1-of-2 watch resets) XAU +7.95% (w~10.4%, 4th cons pos, +0.82%) NDX
+2.46% SX5E +0.55% 000688 +0.77% SOX +0.24%; losers US10Y -4.50% (w~9.2%,
-0.41% MAIN DRAG, 2nd cons neg after +0.14% reset) WTI -8.49% (w~2.75%,
-0.23% contained by the v36 x0.50 cut) SPX -2.27% (w~8.7%, 2nd cons neg)
N225 -0.24%; stale-guard frozen names pnl 0: 000300.SH/HSI/BTC/ETH/CN10Y
(~33% book neutral rank); COPPER/XAU momentum winners offset bond selloff;
v36 WTI cut containment worked (-8.49% at w 2.75% vs -14.55% at w 5.6%).
v37 changes (evidence from the 12-02..12-16 block; v36 watches fired):
  1. US10Y x0.70 -> x0.60 (2ND CONS NEG WITH LARGE PRINT: -1.39%/-4.50%; the
     -4.50% 10d bond selloff is a large print; mirrors the v25-era US10Y
     x1.00->x0.85 2nd-cons-neg -4.07%/-3.36% precedent; v34 chain had said
     x0.60 on 4th cons neg - this fires early on the large print).
  2. SPX x0.65 -> x0.55 (2ND CONS NEG AFTER RE-BOOST: -0.90%/-2.27% following
     the v34 re-boost on 2 cons pos +1.30%/+11.6%; mirrors the v26-era
     SPX x0.65->x0.55 2nd-cons-neg de-boost precedent; prints mild, no <-8%).
  3. 000688.SH x0.70 -> x0.85 (RE-BOOST FIRED: 2 cons pos +10.80%/+0.77%;
     v36 watch said x0.85 on 2 cons pos; keep the air-pocket-after-re-boost
     mirror watch: cut x0.70 on large <-8% or 2nd cons neg after the re-boost).
Kept: WTI x0.50 (2nd cons neg -14.55%/-8.49% but -8.49% NOT <-10% and not
3rd cons; deepen x0.40 only on another large <-10% or 3rd cons neg; re-boost
x0.65 after 2 stable blocks), COPPER x1.00 (+16.69% rebound; 1-of-2 large
watch resets; cut x0.85 on another large <-8% or 2nd cons large), XAU x1.00
(+7.95% 4th cons pos, at max neutral; de-boost x0.85 on 2nd cons neg or large
<-8%), NDX x0.65 (1 pos +2.46% after 1 neg; re-boost x0.75 on 3rd cons pos),
N225 x0.85 (mild -0.24%, 2nd cons neg but no large/<-6%; cut x0.70 on 2nd
cons large or <-6%), SOX x0.35 (1 pos +0.24% after 1 neg; x0.30 on 3rd cons
neg or another large <-10%), SX5E x0.85 (1 pos +0.55% after 1 neg; x0.70 on
2nd cons neg or <-8%), CN10Y x0.70 / BTC x0.15 / ETH x0.75 (frozen stale),
CAP 0.12, SPREAD 0.06, vol exp 0.6, stale-guard, 10-day cadence.
New watches: US10Y (x0.60->x0.50 on 3rd cons neg or another large <-4%;
re-boost x0.70 on 2 cons pos), SPX (x0.55->x0.45 on 3rd cons neg or large
<-8%; re-boost x0.65 on 2 cons pos), 000688 (x0.70 on large <-8% or 2nd cons
neg after re-boost - air-pocket mirror), WTI (x0.40 on another large <-10% or
3rd cons neg; re-boost x0.65 after 2 stable), COPPER (x0.85 on another large
<-8% or 2nd cons large), XAU (x0.85 on 2nd cons neg or large <-8%), NDX
(x0.75 on 3rd cons pos), N225 (x0.70 on 2nd cons large or <-6%), SOX (x0.30
on 3rd cons neg or another large <-10%), SX5E (x0.70 on 2nd cons neg or
<-8%), BTC/CN10Y/ETH/000300/HSI frozen stale-guard watch.

'''
i = content.find('"""')
content = content[:i] + v37_doc + content[i:]

old_mult = '''    "XAU": 1.00, "US10Y": 0.70, "CN10Y": 0.70,   # safe havens (v35: XAU x0.85->x1.00 RE-BOOST 2 cons pos +5.2%/+0.26%; US10Y kept x0.70 single +0.14% pos clears 4th-cons-neg watch; CN10Y kept x0.70 frozen stale)
    "SOX": 0.35, "NDX": 0.65, "ETH": 0.75, "WTI": 0.50, "BTC": 0.15, "N225": 0.85,  # high-beta (v36: WTI x0.65->x0.50 LARGE AIR-POCKET -14.55% after +27.43% stable block, v34 chain fires; SOX kept x0.35 2nd cons neg -7.43%/-3.35% no <-10%; NDX kept x0.65 single -5.31% after 2 pos; BTC/ETH kept frozen stale; N225 kept x0.85 single -5.30% <-6% not hit)
    "SX5E": 0.85, "SPX": 0.65, "000688.SH": 0.70,  # v35: 000688 x0.85->x0.70 LARGE AIR-POCKET -10.20% after +9.2% re-boost (mirror v24/v31 precedent); SPX kept x0.65 2nd cons pos +11.6%/+5.35%; SX5E kept x0.85 1st pos +2.92%'''
new_mult = '''    "XAU": 1.00, "US10Y": 0.60, "CN10Y": 0.70,   # safe havens (v37: US10Y x0.70->x0.60 2nd cons neg -1.39%/-4.50% LARGE print bond selloff; XAU kept x1.00 4th cons pos +7.95%; CN10Y kept x0.70 frozen stale)
    "SOX": 0.35, "NDX": 0.65, "ETH": 0.75, "WTI": 0.50, "BTC": 0.15, "N225": 0.85,  # high-beta (v37: WTI kept x0.50 2nd cons neg -14.55%/-8.49%, -8.49% not <-10%; SOX kept x0.35 1 pos +0.24% after 1 neg; NDX kept x0.65 1 pos +2.46% after 1 neg; BTC/ETH kept frozen stale; N225 kept x0.85 mild -0.24%)
    "SX5E": 0.85, "SPX": 0.55, "000688.SH": 0.85,  # v37: SPX x0.65->x0.55 2nd cons neg -0.90%/-2.27% after re-boost; 000688 x0.70->x0.85 RE-BOOST 2 cons pos +10.80%/+0.77% (keep air-pocket-after-re-boost watch); SX5E kept x0.85 1 pos +0.55% after 1 neg'''

assert old_mult in content, 'mult block not found'
content = content.replace(old_mult, new_mult)

with open('strategy.py', 'w') as f:
    f.write(content)
print('strategy.py updated to v37; lines:', len(content.splitlines()))
