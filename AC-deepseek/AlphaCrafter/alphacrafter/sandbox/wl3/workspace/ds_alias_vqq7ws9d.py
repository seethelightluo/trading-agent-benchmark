
lines = open('memory.txt').readlines()
print("total lines:", len(lines))
for i in range(max(0, len(lines)-30), len(lines)):
    print(i+1, lines[i].rstrip()[:500])
