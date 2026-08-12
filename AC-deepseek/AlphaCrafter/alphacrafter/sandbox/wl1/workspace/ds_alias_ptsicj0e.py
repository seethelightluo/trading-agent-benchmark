lines = open('memory.txt').readlines()
print("total lines:", len(lines))
for i in range(65, len(lines)):
    print("LINE", i+1, ":", lines[i][:300])
