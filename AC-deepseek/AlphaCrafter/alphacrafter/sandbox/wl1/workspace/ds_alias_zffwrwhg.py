import re
with open('memory.txt') as f:
    content = f.read()
# Find all cycle headers
headers = re.findall(r'^(\d{8}) cycle', content, re.M)
print("Last 10 cycle headers:", headers[-10:])
# Find position of last entry
lines = content.split('\n')
print("Total lines:", len(lines))
# print last 3 lines fully
for l in lines[-3:]:
    print("LINE:", l[:500])