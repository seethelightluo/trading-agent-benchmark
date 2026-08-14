from pathlib import Path

p = Path('strategy.py')
src = p.read_text()

# 1) Add SPX_CAP constant after TECH_ASSETS line
anchor1 = 'TECH_ASSETS = ["NDX", "SOX", "000688.SH"]   # live US/China tech complex\n'
add1 = anchor1 + 'SPX_CAP = 0.12           # 2035-03-29 trader re-tune: SPX 3rd consecutive negative block (-4.44%, -9.71%, -2.46%); r21 -11.9%, r60 -23.8%; spx_beta_60 kept pushing SPX to ~16% largest weight -> hard cap 0.12 applied after guard stack\n'
assert anchor1 in src
src = src.replace(anchor1, add1)

# 2) Insert spx_guard function before is_block_start
anchor2 = 'def is_block_start():\n'
guard_fn = '''def spx_guard(w, assets, live, cap=CAP, spx_cap=SPX_CAP):
    """Trader guard (2035-03-29): cap SPX weight at SPX_CAP.

    Trigger met: SPX logged 3 consecutive negative blocks (-4.44% 02-15..03-01,
    -9.71% 03-01..03-15, -2.46% 03-15..03-29) while spx_beta_60(+0.14) kept
    pushing SPX to the portfolio's largest weight (~16%) right before each
    decline. Applied AFTER commodity_guard and tech_guard so the cap acts on
    the true final weights; freed weight water-fills to remaining live assets
    (per-asset cap preserved). Reassess (relax to 0.14) if the Screener
    re-tilts spx_beta_60 down or SPX trend turns positive.
    """
    w = dict(w)
    for _ in range(300):
        excess = 0.0
        for a in assets:
            c = cap
            if a == "SPX" and a in live:
                c = min(c, spx_cap)
            if w[a] > c:
                excess += w[a] - c
                w[a] = c
        if excess < 1e-12:
            break
        room = [a for a in assets if a in live and w[a] < cap - 1e-9]
        if not room:
            break
        p = {a: max(w[a], 1e-9) for a in room}
        den = sum(p.values())
        if den <= 0:
            break
        for a in room:
            w[a] += excess * p[a] / den
    tot = sum(w.values())
    if tot <= 0:
        w = {a: 1.0 / len(assets) for a in assets}
    else:
        w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())  # float guard
    return {a: max(0.0, float(x)) for a, x in w.items()}


'''
assert anchor2 in src
src = src.replace(anchor2, guard_fn + anchor2)

# 3) Apply spx_guard after tech_guard in the hook + print
anchor3 = '    weights = tech_guard(weights, assets, live)\n'
add3 = anchor3 + '    weights = spx_guard(weights, assets, live)\n'
assert anchor3 in src
src = src.replace(anchor3, add3)

anchor4 = '    techw = sum(weights[a] for a in TECH_ASSETS if a in live)\n'
add4 = anchor4 + '    print(f"[trader] SPX cap: SPX={weights[\'SPX\'] * 100:.1f}% (cap {SPX_CAP * 100:.0f}%)")\n'
assert anchor4 in src
src = src.replace(anchor4, add4)

p.write_text(src)
print("patched OK")
