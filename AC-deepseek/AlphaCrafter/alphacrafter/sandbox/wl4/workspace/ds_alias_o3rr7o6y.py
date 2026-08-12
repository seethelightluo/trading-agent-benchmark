import json, os
# Check most recent cycle note in memory around 20281009 / 20281023
with open('memory.txt') as f:
    lines = f.readlines()
# find lines mentioning 202810
for i, l in enumerate(lines):
    if '202810' in l:
        print(i, l[:200])
