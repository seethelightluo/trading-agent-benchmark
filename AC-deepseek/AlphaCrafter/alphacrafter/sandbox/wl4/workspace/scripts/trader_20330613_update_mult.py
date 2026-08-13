from pathlib import Path

p = Path("strategy.py")
s = p.read_text()

# v51 updates to DEFENSIVE_MULT values
s = s.replace('"SOX": 0.20, "NDX": 0.75,', '"SOX": 0.20, "NDX": 0.60,')
s = s.replace('"SX5E": 0.60, "SPX": 0.75, "000688.SH": 0.60,', '"SX5E": 0.70, "SPX": 0.75, "000688.SH": 0.70,')

# refresh the inline comments to v51 status
s = s.replace(
    '# safe havens (v49: XAU kept x1.00 1st neg -2.25% after 2-pos re-boost, x0.85 on 2nd cons neg or <-5%; US10Y kept x0.30 mild neg -0.22%, x0.25 on 2nd cons neg; CN10Y kept x0.70 frozen stale)',
    '# safe havens (v51: XAU kept x1.00 3rd cons pos +3.26% at cap, x0.85 on 2nd cons neg or <-5%; US10Y kept x0.30 1st neg -0.83% after 2-pos run, re-boost x0.35 on 2 cons pos; CN10Y kept x0.70 frozen stale)')
s = s.replace(
    '# high-beta (v49: N225 x0.70->x0.60 ANOTHER large -7.65% after -4.58%/-6.41% chain; WTI x0.30->x0.20 2nd cons neg + large -9.17%; SOX kept x0.20 pos +9.36% resets neg-count, re-boost x0.30 on 2 cons pos; NDX kept x0.75 pos +12.24%, re-boost x1.00 on 2 cons pos; BTC/ETH frozen stale)',
    '# high-beta (v51: NDX x0.75->x0.60 3rd cons neg cum ~-8.7% incl -4.75% MAIN DRAG; WTI kept x0.15 STRONG pos +17.37% 1st after 4-cons-neg air-pocket chain, re-boost x0.20 on 2 stable; SOX kept x0.20 pos +9.83% 1st after negs, re-boost x0.30 on 2 cons pos; N225 kept x0.60 1st mild neg -1.05% after 2-pos run, re-boost x0.70 on 2 cons pos; BTC/ETH frozen stale)')
s = s.replace(
    '# v49: SX5E kept x0.50 pos +5.86%, re-boost x0.60 on 2 cons pos; SPX kept x0.75 1st neg -5.93% after reset, x0.65 on 2nd cons neg or <-8%; 000688 kept x0.60 1st neg -6.93% after 2-pos re-boost, x0.45 on 2nd cons neg or <-8%',
    '# v51: SX5E x0.60->x0.70 3rd cons pos +7.96% TOP winner; SPX kept x0.75 3rd cons pos +0.95%, x0.65 on 2nd cons neg or <-8%, re-boost x0.85 on 2 more cons pos; 000688 x0.60->x0.70 3rd cons pos (+1.25%/+21.41%/+0.12%), x0.45 on 2nd cons neg or <-8%')

p.write_text(s)
print("DEFENSIVE_MULT updated to v51")
# verify
import re
m = re.search(r"DEFENSIVE_MULT = \{(.*?)\}", s, re.S)
print(m.group(0)[:600] if m else "not found")
