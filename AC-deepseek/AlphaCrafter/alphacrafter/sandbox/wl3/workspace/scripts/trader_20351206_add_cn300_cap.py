with open('strategy.py') as f:
    c = f.read()

# 1) add CN300_CAP constant after XAU_CAP line
old_const = 'XAU_CAP = 0.16           # 2035-06-07 trader re-tune: XAU 3rd consecutive negative block at max weight (-0.37%, -1.76%, -3.54%, -6.88%); recurring biggest drag -> hard cap 0.18->0.16 applied after guard stack'
new_const = old_const + '\nCN300_CAP = 0.14          # 2035-12-06 trader re-tune: 000300.SH 3rd consecutive negative block at max weight (10-25 -5.71% at 16.5%, 11-08 -0.54%, 11-22 -2.88% at ~17% largest position); plan trigger fired -> hard cap 0.18->0.14 applied after guard stack'
assert old_const in c, 'const anchor missing'
c = c.replace(old_const, new_const)

# 2) add param to apply_all_caps signature
old_sig = 'def apply_all_caps(w, assets, live, stress=False, cap=CAP, spx_cap=SPX_CAP,\n                     xau_cap=XAU_CAP, wti_cap=WTI_CAP, copper_cap=COPPER_CAP,'
new_sig = 'def apply_all_caps(w, assets, live, stress=False, cap=CAP, spx_cap=SPX_CAP,\n                     xau_cap=XAU_CAP, cn300_cap=CN300_CAP, wti_cap=WTI_CAP, copper_cap=COPPER_CAP,'
assert old_sig in c, 'sig anchor missing'
c = c.replace(old_sig, new_sig)

# 3) add cfor branch
old_cfor = '        if a == "XAU":\n            c = min(c, xau_cap)\n        if a == "WTI":'
new_cfor = '        if a == "XAU":\n            c = min(c, xau_cap)\n        if a == "000300.SH":\n            c = min(c, cn300_cap)\n        if a == "WTI":'
assert old_cfor in c, 'cfor anchor missing'
c = c.replace(old_cfor, new_cfor)

# 4) add print in hook after SPX cap print
old_print = '    print(f"[trader] SPX cap: SPX={weights[\'SPX\'] * 100:.1f}% (cap {SPX_CAP * 100:.0f}%)")'
new_print = old_print + '\n    print(f"[trader] 000300 cap: 000300.SH={weights[\'000300.SH\'] * 100:.1f}% (cap {CN300_CAP * 100:.0f}%)")'
assert old_print in c, 'print anchor missing'
c = c.replace(old_print, new_print)

# 5) docstring guard list update
old_doc = 'SPX cap 0.12 (2035-03-29), XAU cap 0.16 (2035-06-07).'
new_doc = 'SPX cap 0.12 (2035-03-29), XAU cap 0.16 (2035-06-07),\n000300.SH cap 0.14 (2035-12-06).'
assert old_doc in c, 'doc anchor missing'
c = c.replace(old_doc, new_doc)

with open('strategy.py', 'w') as f:
    f.write(c)
print('edits applied OK')
