import json

import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import cm
from matplotlib import style
from matplotlib.pyplot import figure
import numpy as np
import cv2
from colorspacious import cspace_converter
plt.rcParams['text.usetex'] = True

res = 200

legend_font_size = 14
axis_font_size = 20
tickle_font_size = 18
size = (6, 6)

n_clusters = 9

## Eigenvalues Plot
#

x_val = [i for i in range(32)]

eigenvalues_json = open('./metrics/eigenvalues.json')
eigenvalues_data = json.load(eigenvalues_json)
eigenvalues_data.reverse()
plt.figure(figsize=size, dpi=res)
plt.bar(x_val, eigenvalues_data)
plt.xticks(fontsize=tickle_font_size)
plt.yticks(fontsize=tickle_font_size)
plt.title('Eigenvalues: SpRAy analysis of ResNet18', fontsize=axis_font_size)
#plt.show()
plt.savefig('./plots/eigenvalues.png',dpi=res, bbox_inches = "tight")
plt.clf()

## TSNE
#
tsne_json = open('./metrics/tsne_0.json')
tsne_data = json.load(tsne_json)

X = []
Y = []

for point in tsne_data:
    X.append(point[0])
    Y.append(point[1])

plt.figure(figsize=size, dpi=res)
plt.xticks(fontsize=tickle_font_size)
plt.yticks(fontsize=tickle_font_size)
plt.title('t-SNE plot', fontsize=axis_font_size)
plt.scatter(X, Y, s=10)
#plt.show()
plt.savefig('./plots/tsne_points.png',dpi=res, bbox_inches = "tight")
plt.clf()

## Clustering
#
kmeans_json = open('./metrics/kmeans_0.json')
kmeans_data = json.load(kmeans_json)
print((kmeans_data[0][0]))

mask = kmeans_data[n_clusters-2]

plt.figure(figsize=size, dpi=res)
plt.xticks(fontsize=tickle_font_size)
plt.yticks(fontsize=tickle_font_size)
#ax = plt.gca()
#ax.axes.xaxis.set_visible(False)
#ax.axes.yaxis.set_visible(False)
plt.title('K-Means: cluster size = ' + str(n_clusters), fontsize=axis_font_size)

heatmaps_json = open('./metrics/hw_data.json')
heatmaps_data = json.load(heatmaps_json)

cluster_heatmaps = []

for cluster_idx in range(n_clusters):
    X_cluster = []
    Y_cluster = []
    heatmaps = []
    for mask_idx, x, y, heatmap in zip(mask, X, Y, heatmaps_data):
        if mask_idx == cluster_idx:
            X_cluster.append(x)
            Y_cluster.append(y)
            heatmaps.append(heatmap)
    plt.scatter(X_cluster, Y_cluster, s=10)
    print("Cluster ", str(cluster_idx), ": ", len(heatmaps))
    cluster_heatmaps.append(heatmaps)

plt.savefig('./plots/kmeans_cluster_' + str(n_clusters) + '.png',dpi=res, bbox_inches = "tight")
plt.clf()

print(len(cluster_heatmaps))

for idx in range(n_clusters):
    image_data = []
    for img in cluster_heatmaps[idx]:
        this_image = cv2.imread("lrp_results/handwheels_attrib/" + img)
        image_data.append(this_image)

    avg_image = image_data[0]
    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)
    normalized = cv2.normalize(avg_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite('./plots/mean_map_' + str(idx) + '.jpg', normalized)
