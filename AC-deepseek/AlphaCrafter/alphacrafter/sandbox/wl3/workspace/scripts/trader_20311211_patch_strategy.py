"""Patch strategy.py: 2031-12-11 cap re-tune (per 11-27 plan trigger: commodities bleeding).

Changes:
1. COMM_CAP 0.36 -> 0.33 (XAU+COPPER+WTI complex)
2. NEW COPPER_CAP = 0.12 single-name cap (COPPER was ~15% and the largest block drag)
3. ETH cap 6% now UNCONDITIONAL (was stress-only in risk_trim); applied in
   commodity_guard so it acts on final weights in all regimes.
"""
import re
from pathlib import Path

p = Path("strategy.py")
src = p.read_text()

# 1. COMM_CAP constant
src = src.replace(
    "COMM_CAP = 0.36          # XAU+COPPER+WTI combined cap (defensive XAU kept, cyclicals limited)",
    "COMM_CAP = 0.33          # XAU+COPPER+WTI combined cap (trimmed 0.36->0.33 on 2031-12-11)\nCOPPER_CAP = 0.12          # 2031-12-11 single-name COPPER cap (COPPER ~15% was main block drag)\nETH_CAP_ALL = 0.06         # 2031-12-11: ETH cap made unconditional (2 consecutive crypto-loss blocks)",
)

# 2. commodity_guard signature
src = src.replace(
    "def commodity_guard(w, assets, live, cap=CAP, wti_cap=WTI_CAP, comm_cap=COMM_CAP):",
    "def commodity_guard(w, assets, live, cap=CAP, wti_cap=WTI_CAP,\n                       comm_cap=COMM_CAP, copper_cap=COPPER_CAP, eth_cap=ETH_CAP_ALL):",
)

# 3. commodity_guard loop: cap WTI, COPPER, ETH
src = src.replace(
    """        for a in assets:
            c = cap
            if a == "WTI" and a in live:
                c = min(c, wti_cap)
            if w[a] > c:
                excess += w[a] - c
                w[a] = c
        s_comm = sum(w[a] for a in comm)""",
    """        for a in assets:
            c = cap
            if a == "WTI" and a in live:
                c = min(c, wti_cap)
            if a == "COPPER" and a in live:
                c = min(c, copper_cap)
            if a == "ETH" and a in live:
                c = min(c, eth_cap)
            if w[a] > c:
                excess += w[a] - c
                w[a] = c
        s_comm = sum(w[a] for a in comm)""",
)

# 4. room logic: ETH also excluded once capped
src = src.replace(
    """        room = []
        for a in assets:
            if a not in live:
                continue
            c = cap
            if a == "WTI":
                c = min(c, wti_cap)
            if w[a] < c - 1e-9:
                room.append(a)""",
    """        room = []
        for a in assets:
            if a not in live:
                continue
            c = cap
            if a == "WTI":
                c = min(c, wti_cap)
            if a == "COPPER":
                c = min(c, copper_cap)
            if a == "ETH":
                c = min(c, eth_cap)
            if w[a] < c - 1e-9:
                room.append(a)""",
)

# 5. strategy_hook print: add COPPER cap + ETH unconditional note
src = src.replace(
    '    print(f"[trader] commodity guard: XAU={weights[\'XAU\'] * 100:.1f}% "\n'
    '          f"COPPER={weights[\'COPPER\'] * 100:.1f}% WTI={weights[\'WTI\'] * 100:.1f}% "\n'
    '          f"complex={sum(weights[a] for a in (\'XAU\',\'COPPER\',\'WTI\')) * 100:.1f}% "\n'
    '          f"(WTI cap {WTI_CAP * 100:.0f}%, complex cap {COMM_CAP * 100:.0f}%)")',
    '    print(f"[trader] commodity guard: XAU={weights[\'XAU\'] * 100:.1f}% "\n'
    '          f"COPPER={weights[\'COPPER\'] * 100:.1f}% WTI={weights[\'WTI\'] * 100:.1f}% "\n'
    '          f"complex={sum(weights[a] for a in (\'XAU\',\'COPPER\',\'WTI\')) * 100:.1f}% "\n'
    '          f"ETH={weights[\'ETH\'] * 100:.1f}% "\n'
    '          f"(WTI cap {WTI_CAP * 100:.0f}%, COPPER cap {COPPER_CAP * 100:.0f}%, "\n'
    '          f"complex cap {COMM_CAP * 100:.0f}%, ETH cap {ETH_CAP_ALL * 100:.0f}%)")',
)

# 6. module docstring: append re-tune note
src = src.replace(
    '2031-04-03 COMMODITY GUARD (trader, ensemble unchanged after Screener escalation): WTI_CAP=0.06, COMM_CAP=0.36 (XAU+COPPER+WTI) applied after risk_trim. Rationale: WTI -14% block (03-20..04-03), 3rd consecutive negative commodity-beta attribution.',
    '2031-04-03 COMMODITY GUARD (trader, ensemble unchanged after Screener escalation): WTI_CAP=0.06, COMM_CAP=0.36 (XAU+COPPER+WTI) applied after risk_trim. Rationale: WTI -14% block (03-20..04-03), 3rd consecutive negative commodity-beta attribution.\n\n2031-12-11 CAP RE-TUNE (trader, per 11-27 plan trigger: commodities kept bleeding -\nWTI -16.5%/10d accelerating, COPPER -3.5%/10d; backtest Sharpe recovered to 0.38):\nCOMM_CAP 0.36->0.33, NEW COPPER_CAP=0.12 (COPPER ~15% was largest block drag),\nETH cap 6% made UNCONDITIONAL (2 consecutive crypto-loss blocks: -15.8%, -3.7%).\nApplied in commodity_guard (final-weight layer) so it holds in all regimes.',
)

p.write_text(src)
print("patched OK")
# sanity: show key lines
for line in src.splitlines():
    if any(k in line for k in ("COPPER_CAP", "COMM_CAP =", "ETH_CAP_ALL", "copper_cap", "eth_cap=eth_cap", "a == \"COPPER\"", "a == \"ETH\"")):
        print(">", line)
