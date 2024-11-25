import torch
import numpy as np
import matplotlib.pyplot as plt

def feature_color(feature):
    feature = feature[0][0]
    dim = feature.shape[0]
    feature_norm = torch.sum(feature, 0) / dim
    return feature_norm.data.cpu().numpy()
    

feature = torch.load('1.pt')
feature360 = torch.load('360.pt')
processed = []
processed.append(feature_color(feature))
processed.append(feature_color(feature360))

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")

plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')