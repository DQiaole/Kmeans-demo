import warnings
warnings.filterwarnings('ignore')

import numpy as np
from time import time
import argparse
from copy import deepcopy
from utils import get_accuracy, randCent, kmeans_plus_plus_Cent
from kmeans_func import k_means_hist as k_means

#dataset loading module
from sklearn.datasets import load_svmlight_file, load_iris
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

def main(args):
    n_run = args.n_run
    if args.dataset == "MNIST":
        X_r, y = torch.load('dataset/MNIST/processed/training.pt')
        X = X_r.numpy().reshape(X_r.shape[0],-1)
        y = y.numpy()
    elif args.dataset == 'iris':
        X, y = load_iris(return_X_y=True)
    else:
        print("unknown dataset!!!")
        return 0
    n_centers = len(np.unique(y))
    random_seeds = np.random.randint(1, 1000, n_run).tolist()
    X = StandardScaler().fit_transform(X)
    print("random seed:", random_seeds)
    methods = ["kmeans", "kmeans++"]
    metrics_name = ["seed", "ACC", "ARI", "NMI", "silhouette", "time"]
    results_dict = {}
    for method in methods:
        results_dict[method] = {"method":[method] * n_run}
        for metric in metrics_name:
            results_dict[method][metric] = []

    for random_seed in random_seeds:
        print(random_seed)
        np.random.seed(random_seed)

        centroids_random = randCent(X, n_centers, random_seed)
        centroids_kmeans = kmeans_plus_plus_Cent(X, n_centers, random_seed)
        max_iter = 300
        
        # k-means
        centroids_init = deepcopy(centroids_random)
        start_t = time()
        centroids, clusterAssment, centroids_hist, num_iter, sse_hist, label_hist = k_means(
            X, n_centers, centroids_init, random_state=random_seed, max_iter = max_iter)
        end_t = time()
        total_time = end_t - start_t
        y_pred = clusterAssment[:, 0]
        cluster_center = centroids

        ACC_socre, _ = get_accuracy(y_pred.astype(np.int), y, n_centers)
        ARI_socre = metrics.adjusted_rand_score(y, y_pred.astype(np.int))
        NMI_score = metrics.normalized_mutual_info_score(y, y_pred)
        silhouette_score = metrics.silhouette_score(X, y_pred, metric='euclidean')
        values = [random_seed, ACC_socre, ARI_socre, NMI_score, silhouette_score, total_time]

        print('# Kmeans')
        for metric, value in zip(metrics_name,values):
            print("{}:{}".format(metric, value))
            results_dict["kmeans"][metric].append(value)

        # k-means++
        centroids_init = deepcopy(centroids_kmeans)
        start_t = time()
        centroids, clusterAssment, centroids_hist, num_iter, sse_hist, label_hist = k_means(
            X, n_centers, centroids_init, random_state=random_seed, max_iter = max_iter)
        end_t = time()
        total_time = end_t - start_t
        y_pred = clusterAssment[:, 0]
        cluster_center = centroids
        ACC_socre, _ = get_accuracy(y_pred.astype(np.int), y, n_centers)
        ARI_socre = metrics.adjusted_rand_score(y, y_pred.astype(np.int))
        NMI_score = metrics.normalized_mutual_info_score(y, y_pred)
        silhouette_score = metrics.silhouette_score(X, y_pred, metric='euclidean')
        values = [random_seed, ACC_socre, ARI_socre, NMI_score, silhouette_score, total_time]
        
        print('# Kmeans++')
        for metric, value in zip(metrics_name,values):
            print("{}:{}".format(metric, value))
            results_dict["kmeans++"][metric].append(value)
    
    kmeans_results =pd.DataFrame(results_dict["kmeans"])
    kmeans_results.to_csv("kmeans_results_{}.csv".format(args.dataset), index=False)
    kmeans_pp_results =pd.DataFrame(results_dict["kmeans++"])
    kmeans_pp_results.to_csv("kmeans_plusplus_results_{}.csv".format(args.dataset), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kmeans')
    parser.add_argument('--dataset', default='iris', type=str, help='dataset')
    parser.add_argument('--n_run', default=10, type=int, help='n_run')

    args = parser.parse_args()
    print(args)
    main(args)