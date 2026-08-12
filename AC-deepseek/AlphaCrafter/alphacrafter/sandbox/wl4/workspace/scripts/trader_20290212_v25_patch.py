"""Trader patch: v24 -> v25 (2029-02-12). Update header doc + DEFENSIVE_MULT."""
from pathlib import Path

p = Path("strategy.py")
src = p.read_text()

# ---- 1. Insert v25 header section after the opening docstring ----
anchor = '"""Trader strategy v24 on top of v23'
assert src.count(anchor) == 1, f"anchor count {src.count(anchor)}"
v25_header = '''"""Trader strategy v25 on top of v24 (fired 2029-02-12 after the 01-29..02-12
block -0.04%, DD 0.5% - FLAT through a risk-off tape: BTC -18.9%, CN10Y
-10.4%, COPPER -5.8%, SOX -2.5%, 000688 -2.5%, N225 -0.8%; winners SX5E
+6.3%, WTI +4.8%, XAU +4.5%, NDX +2.2%, SPX +1.5%, US10Y +0.1%; mean asset
-1.42%, portfolio -0.04% => defensive cuts contained the air-pockets; v24
proposal SKIPPED by the deterministic gate (gross edge <= turnover*3bp) ->
holdings unchanged):
  v25 changes (evidence from the 01-29..02-12 block):
    1. XAU x0.85 -> x1.00 (PRE-AGREED RE-BOOST FIRED: 2 consecutive positive
       10d blocks (+~2% 01-01..01-15, +4.50% this block); v24 watch said
       re-boost on 2 consecutive positives; XAU functioning as safe haven
       while BTC/CN10Y crash).
    2. SX5E x0.85 -> x1.00 (RE-BOOST FIRED: 2 consecutive positive blocks
       (+~2% 01-01..01-15, +6.28% this block); strongest relative performer).
    3. NDX x0.65 -> x0.75 (RE-BOOST FIRED: 2 consecutive positive blocks
       (top winner 01-01..01-15, +2.17% this block); v16/v22 2-stable-block
       re-boost precedent).
    4. BTC x0.40 -> x0.35 (PRE-AGREED DEEPEN FIRED: 5th consecutive negative
       block (-1.20%/-3.95%/-7.96%/-small/-18.91%) with a large -18.91%
       print; v24 watch said deepen to x0.35 on 4th consecutive negative;
       drag still ~-0.35% of book at x0.40).
    5. CN10Y x0.70 -> x0.60 (REPEAT-OFFENDER DEEPEN FIRED: 3 large prints in
       4 traded blocks -10.42% (10-09..10-23) / -7.91% (10-23..11-06) /
       -10.37% (01-29..02-12); mirrors the SOX repeat-offender escalation
       chain; rate_beta_cn10y_60d dir=-1 w=0.24 keeps the factor hedge).
  Kept: US10Y x1.25 (+0.12% flat but safe haven), SOX x0.55 (mild -2.52%),
  ETH x0.75 (frozen), WTI x0.65 (+4.84% single positive after -12.79% large
  print -> needs 2 stable blocks before re-boost), SPX x0.65 (+1.52% single
  positive after 4 negatives -> re-boost needs 2 consecutive positives),
  000688.SH x0.85 (-2.51% single -> watch: cut to x0.70 on 2nd consecutive
  negative or large print), CAP 0.12, SPREAD 0.06, vol exp 0.6, stale-guard.
  New watches: COPPER (single -5.75% after +4.85% -> cut only on 2nd
  consecutive large), N225 (mild -0.76% -> large-print watch), BTC deepen
  (6th consecutive negative -> x0.30), CN10Y deepen (another large print or
  2nd consecutive negative -> x0.50), WTI re-boost (2 stable blocks ->
  x0.80), SPX re-boost (2 consecutive positives -> x0.85).

Trader strategy v24 on top of v23'''
src = src.replace(anchor, v25_header, 1)

# ---- 2. Insert v25 comment above the v24 comment block ----
old_c = "# v24 defensive multipliers (fired 2028-11-06). Prior v23 (2028-10-23):"
assert src.count(old_c) == 1
new_c = """# v25 defensive multipliers (fired 2029-02-12 after the 01-29..02-12 block -0.04%):
#   XAU x0.85 -> x1.00 (re-boost, 2 cons positives), SX5E x0.85 -> x1.00 (re-boost),
#   NDX x0.65 -> x0.75 (re-boost, 2 cons positives), BTC x0.40 -> x0.35 (5th cons neg,
#   -18.91% large print), CN10Y x0.70 -> x0.60 (repeat-offender 3 large prints)
# Prior v24 (fired 2028-11-06). Prior v23 (2028-10-23):"""
src = src.replace(old_c, new_c, 1)

# ---- 3. Replace DEFENSIVE_MULT block ----
lines = src.split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith("DEFENSIVE_MULT = {"))
end = start + 1
while not lines[end].lstrip().startswith("}"):
    end += 1
new_block = '''DEFENSIVE_MULT = {
    "XAU": 1.00, "US10Y": 1.25, "CN10Y": 0.60,   # safe havens (v25: XAU x0.85->x1.00 re-boost 2 cons positives; CN10Y x0.70->x0.60 repeat-offender 3 large prints -10.42%/-7.91%/-10.37%; US10Y kept x1.25)
    "SOX": 0.55, "NDX": 0.75, "ETH": 0.75, "WTI": 0.65, "BTC": 0.35,  # high-beta cuts (v25: NDX x0.65->x0.75 re-boost 2 cons positives; BTC x0.40->x0.35 5th cons neg incl -18.91% large; SOX/ETH/WTI kept)
    "SX5E": 1.00, "SPX": 0.65, "000688.SH": 0.85,  # v25: SX5E x0.85->x1.00 re-boost 2 cons positives incl +6.28%; SPX kept x0.65 (re-boost needs 2 cons); 000688 kept x0.85 (single -2.51% watch)
}'''
lines = lines[:start] + new_block.split("\n") + lines[end + 1:]
src = "\n".join(lines)
p.write_text(src)
print("PATCH OK")
