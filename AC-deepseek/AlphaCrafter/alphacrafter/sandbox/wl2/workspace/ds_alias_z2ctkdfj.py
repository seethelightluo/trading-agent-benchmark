with open('memory.txt') as f:
    lines = f.readlines()
print("total lines:", len(lines))
print("--- FIRST 3 ---")
print("".join(lines[:3]))
print("--- LAST 3 ---")
print("".join(lines[-3:]))
# find entries mentioning 2034/2035
import re
recent = [l for l in lines if re.search(r'203[45]', l)]
print("entries mentioning 2034/2035:", len(recent))
print("".join(recent[-6:]))