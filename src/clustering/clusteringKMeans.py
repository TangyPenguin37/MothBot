import csv
import numpy as np
from sklearn.cluster import KMeans

def kmeans_clustering(data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_, kmeans.n_iter_

# use all columns except the first one (id)
csvData = np.loadtxt(
    'data/unlabelled_batch1/data_formatted.csv',
    delimiter=',',
    skiprows=1,
    usecols=range(1, 193))
    # usecols=[1, 5])

labels, centers, inertia, n_iter = kmeans_clustering(csvData, n_clusters=2)

# print(labels.shape)
print(centers)

with open('temp.csv', 'w', newline='', encoding='UTF8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(centers[0])
    writer.writerow(centers[1])

# # plot
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # plot example data
# ax.scatter(csvData[:, 0], csvData[:, 1], csvData[:, 2], c=labels, cmap='viridis', linewidth=0.5)

# # plot centers
# ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5)

# # plot labels
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# plt.show()