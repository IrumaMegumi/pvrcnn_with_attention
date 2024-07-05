mlps=[[16,16],[16,16]]
input_channels=5
for k in range(len(mlps)):
    mlps[k] = [input_channels] + mlps[k]
print("yes")