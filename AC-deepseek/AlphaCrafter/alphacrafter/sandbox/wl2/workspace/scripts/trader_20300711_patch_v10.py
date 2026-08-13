"""Patch strategy.py with v10 commodity parabolic-move gate (cycle 78, 2030-07-11).

Evidence: cycles 75-77 commodity momentum rewards then inverts violently -
COPPER cap-add -7.7% (top drag) and WTI add -2.9% after prior-block surges;
WTI now +25.9% r20 (parabolic) again at decision date. Gate: cap WTI/COPPER at
.09 when live 20d return > +15%. Small targeted change; rest of v6 untouched.
"""
from pathlib import Path

p = Path("strategy.py")
src = p.read_text()

# 1) constants after MOM_TREND_CAP
anchor_const = "MOM_TREND_CAP = 0.09\n"
add_const = (
    "MOM_TREND_CAP = 0.09\n"
    "COMMODITY_PARABOLIC = {\"WTI\", \"COPPER\"}   # commodity momentum inversion guard (cycles 75-77 evidence; v10 2030-07-11)\n"
    "COMMODITY_PARABOLIC_THRESH = 0.15          # 20d return above which a commodity surge is 'parabolic'\n"
    "COMMODITY_PARABOLIC_CAP = 0.09\n"
)
assert src.count(anchor_const) == 1, "constants anchor not unique"
src = src.replace(anchor_const, add_const)

# 2) gate loop after the MOMENTUM_ADD loop in build_target
anchor_loop = """    for a in MOMENTUM_ADD:
        if r20[a] < MOM_TREND_THRESH:
            cap_map[a] = min(cap_map.get(a, CAP), MOM_TREND_CAP)
"""
add_loop = anchor_loop + """    for a in COMMODITY_PARABOLIC:
        if r20[a] > COMMODITY_PARABOLIC_THRESH:
            cap_map[a] = min(cap_map.get(a, CAP), COMMODITY_PARABOLIC_CAP)
"""
assert src.count(anchor_loop) == 1, "loop anchor not unique"
src = src.replace(anchor_loop, add_loop)

# 3) docstring header note (append after v9 line, first occurrence)
anchor_doc = "momentum-add gate (SOX/N225 capped at 0.09 when 20d return < -2%) added cycle57"
add_doc = ("momentum-add gate (SOX/N225 capped at 0.09 when 20d return < -2%) added cycle57; v10\n"
           "commodity parabolic-move gate (WTI/COPPER capped at 0.09 when live 20d return > +15%)\n"
           "added cycle78 - commodity momentum rewards then inverts (cycles 75-77: COPPER -7.7% /\n"
           "WTI -2.9% add regrets after prior-block surges).")
assert src.count(anchor_doc) == 1, "doc anchor not unique"
src = src.replace(anchor_doc, add_doc)

p.write_text(src)
print("patched OK")
