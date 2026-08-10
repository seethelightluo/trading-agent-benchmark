src7 = open('scripts/miner3_20260716_screen_cycle7.py').read()
src7b = open('scripts/miner3_20260716_screen_cycle7b.py').read()
import re
c7 = set(re.findall(r'cands\["([^"]+)"\]', src7))
c7b = set(re.findall(r'cands\["([^"]+)"\]', src7b))
print("cycle7 only:", sorted(c7 - c7b))
print("cycle7b only:", sorted(c7b - c7))
print("common:", len(c7 & c7b))
# check syntax of both
import ast
for name, src in [('cycle7', src7), ('cycle7b', src7b)]:
    try:
        ast.parse(src)
        print(name, "OK syntax")
    except SyntaxError as e:
        print(name, "SYNTAX ERR", e)
EOF