with open('memory.txt') as f:
    lines = f.readlines()
print(f"total lines: {len(lines)}")
# print first 12 lines
for l in lines[:12]:
    print(l.rstrip()[:500])
