import numpy as np
from copy import deepcopy
from scipy.spatial import distance

def k_means_hist(dataset, k, centroids_init, max_iter=300, distmeas='euclidean', threshold=1e-3, random_state=0):
    np.random.seed(random_state)
    m = dataset.shape[0]
    clusterAssment = np.zeros((m, 2))
    centroids = deepcopy(centroids_init)

    # record centroids (centroids history)
    centroids_hist = [[] for _ in range(k)]
    centroids_cp = deepcopy(centroids_init)
    for j in range(k):
        centroids_hist[j].append(centroids_cp[j, :])

    clusterChanged = True
    num_iter = 0
    sse_hist = []
    label_hist = []
    while clusterChanged:
        # compute distances
        dist = distance.cdist(dataset, centroids, metric=distmeas)
        minIndex = dist.argmin(1)
        minDist = dist.min(1)
        clusterAssment[:, 0], clusterAssment[:, 1] = minIndex, minDist**2
        sse_hist.append(np.sum(minDist**2))
        label_hist.append(minIndex)

        clusterChanged = False
        for j in range(k):
            ptsInClust = dataset[clusterAssment[:, 0] == j]
            new_centroid = np.mean(ptsInClust, axis=0)

            dist_center = distance.cdist(new_centroid.reshape(
                1, -1), centroids[j, :].reshape(1, -1), metric=distmeas)
            if dist_center > threshold:
                centroids[j, :] = new_centroid
                centroids_hist[j].append(new_centroid)
                clusterChanged = True

        num_iter += 1
        if num_iter >= max_iter:
            print("iteration >= {}".format(max_iter))
            break

    return centroids, clusterAssment, np.array(centroids_hist), num_iter, sse_hist, label_hist
