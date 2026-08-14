from pathlib import Path

p = Path('strategy.py')
src = p.read_text()

# 1) Replace the spx_guard function (added 2035-03-29) with a comprehensive
#    final cap guard that enforces ALL sub-caps simultaneously (per-asset 0.18,
#    SPX 0.12, WTI 0.04, COPPER 0.10, ETH 0.04, tech complex 0.24,
#    comm complex 0.33, equity complex 0.40 under stress) with a non-destructive
#    water-fill that preserves the frozen floor and sum-to-1.
start = src.index('def spx_guard(')
end = src.index('def is_block_start():')
new_fn = '''def apply_all_caps(w, assets, live, stress=False, cap=CAP, spx_cap=SPX_CAP,
                     wti_cap=WTI_CAP, copper_cap=COPPER_CAP, eth_cap=ETH_CAP_ALL,
                     tech_cap=TECH_CAP, comm_cap=COMM_CAP, eq_cap=EQ_CAP):
    """Comprehensive final cap guard (2035-04-12).

    Replaces the sequential commodity_guard -> tech_guard -> spx_guard tail.
    The sequential stack let each guard's water-fill redistribute freed weight
    to assets already at their sub-caps (the last guard only knew per-asset
    0.18 + SPX 0.12), so the 03-29 proposal breached COPPER/ETH/WTI/tech/comm
    caps (COPPER 12.9%, ETH 5.2%, WTI 5.2%, tech 26.4%, comm 35%). This guard
    enforces every cap in ONE water-fill loop and does NOT destructively
    renormalize (which had also drifted the frozen floor 0.5% -> ~1%): the
    frozen floor stays at FROZEN_FLOOR and sum-to-1 is restored by filling
    remaining room, not by scaling capped assets above their limits.
    """
    w = dict(w)
    tech = [a for a in TECH_ASSETS if a in live]
    comm = [a for a in ("XAU", "COPPER", "WTI") if a in live]
    eq = [a for a in EQ_ASSETS if a in live]

    def cfor(a):
        c = cap
        if a == "SPX":
            c = min(c, spx_cap)
        if a == "WTI":
            c = min(c, wti_cap)
        if a == "COPPER":
            c = min(c, copper_cap)
        if a == "ETH":
            c = min(c, eth_cap)
        return c

    def room_ok(a, w):
        if w[a] >= cfor(a) - 1e-9:
            return False
        if a in tech and sum(w[x] for x in tech) >= tech_cap - 1e-9:
            return False
        if a in comm and sum(w[x] for x in comm) >= comm_cap - 1e-9:
            return False
        if stress and a in eq and sum(w[x] for x in eq) >= eq_cap - 1e-9:
            return False
        return True

    for _ in range(500):
        excess = 0.0
        for a in assets:
            c = cfor(a)
            if w[a] > c:
                excess += w[a] - c
                w[a] = c
        if stress:
            s_eq = sum(w[a] for a in eq)
            if s_eq > eq_cap:
                excess += s_eq - eq_cap
                for a in eq:
                    w[a] *= eq_cap / max(s_eq, 1e-12)
        s_tech = sum(w[a] for a in tech)
        if s_tech > tech_cap:
            excess += s_tech - tech_cap
            for a in tech:
                w[a] *= tech_cap / max(s_tech, 1e-12)
        s_comm = sum(w[a] for a in comm)
        if s_comm > comm_cap:
            excess += s_comm - comm_cap
            for a in comm:
                w[a] *= comm_cap / max(s_comm, 1e-12)
        if excess < 1e-12:
            break
        room = [a for a in assets if a in live and room_ok(a, w)]
        if not room:
            break
        p = {a: max(w[a], 1e-9) for a in room}
        den = sum(p.values())
        if den <= 0:
            break
        for a in room:
            w[a] += excess * p[a] / den

    # restore exact sum-to-1 by filling remaining room (no destructive scaling)
    tot = sum(w.values())
    diff = 1.0 - tot
    if abs(diff) > 1e-9:
        room = [a for a in assets if a in live and room_ok(a, w)]
        if room and diff > 0:
            p = {a: max(w[a], 1e-9) for a in room}
            den = sum(p.values())
            if den > 0:
                for a in room:
                    w[a] += diff * p[a] / den
        tot = sum(w.values())
        w[assets[-1]] += 1.0 - tot  # float guard (<=1e-12)
    return {a: max(0.0, float(x)) for a, x in w.items()}


'''
src = src[:start] + new_fn + src[end:]

# 2) Hook: replace the three sequential guard calls with the single final guard
old_tail = ('    weights = commodity_guard(weights, assets, live)\n'
            '    weights = tech_guard(weights, assets, live)\n'
            '    weights = spx_guard(weights, assets, live)\n')
new_tail = ('    weights = apply_all_caps(weights, assets, live, stress=stress)\n')
assert old_tail in src, "guard tail not found"
src = src.replace(old_tail, new_tail)

# 3) Update docstring header to mention the final guard
old_hdr = ('Trader guards: frozen-5 pin (0.5% floor), equity-stress trim (eq<=0.40,\\n'
           'ETH<=0.06), commodity guard (WTI<=0.04, COPPER<=0.10, XAU+COPPER+WTI<=0.33,\\n'
           'ETH<=0.04), tech guard (NDX+SOX+000688.SH<=0.24, since 2034-04-27).\\n')
new_hdr = ('Trader guards: frozen-5 pin (0.5% floor), equity-stress trim (eq<=0.40,\\n'
           'ETH<=0.06), commodity guard (WTI<=0.04, COPPER<=0.10, XAU+COPPER+WTI<=0.33,\\n'
           'ETH<=0.04), tech guard (NDX+SOX+000688.SH<=0.24, since 2034-04-27),\\n'
           'SPX cap (<=0.12, since 2035-03-29); all sub-caps enforced together by\\n'
           'apply_all_caps (single final water-fill, no destructive renormalization).\\n')
if old_hdr in src:
    src = src.replace(old_hdr, new_hdr)

p.write_text(src)
print("patched OK")
