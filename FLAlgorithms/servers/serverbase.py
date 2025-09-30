import functools
from collections import defaultdict
import heapq
import math
import hdbscan
import torch
import os
import numpy as np
import h5py
import copy
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from FLAlgorithms.functions.miloss import Mine
import random
import torch.nn.functional as F
import core
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import copy
import time
import hdbscan
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from core.models.Update import LocalUpdate
import heapq
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
import os
import csv
from scipy.linalg import svd
from sklearn.impute import SimpleImputer
# from utils.autoencoder import  ContrastiveAutoencoder,ContrastiveLoss,UnsupervisedContrastiveLoss
from torch.utils.data import TensorDataset,DataLoader
from joblib import load
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import median_abs_deviation
from sklearn.cluster import DBSCAN
from torch.nn.functional import softmax
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from torch.nn import CosineSimilarity
from sklearn.preprocessing import MaxAbsScaler
from sklearn.cluster import AgglomerativeClustering
import random
from scipy import stats
from torch.utils.data import DataLoader, Subset
import torch
from scipy.fftpack import dct
from hdbscan import HDBSCAN


def flatten_weights(weights_dict):
    """Convert a dictionary of tensors into a single flattened numpy array."""
    flattened_weights = []
    for key, tensor in weights_dict.items():
        flattened_weights.append(tensor.flatten().cpu().numpy())
    return np.concatenate(flattened_weights)


def compute_cluster_metrics(flattened_weights):
    """Compute the cluster metrics for each client: model update itself + cosine similarity + Euclidean distance."""
    num_clients = len(flattened_weights)
    cosine_similarity = torch.nn.CosineSimilarity(dim=0)
    metrics = []

    for i in range(num_clients):
        eu_distances = []
        cos_similarities = []

        for j in range(num_clients):
            if i != j:
                # Compute Euclidean distance and cosine similarity between client i and client j
                eu_dist = torch.dist(torch.tensor(flattened_weights[i]), torch.tensor(flattened_weights[j])).item()
                cos_sim = cosine_similarity(torch.tensor(flattened_weights[i]),
                                            torch.tensor(flattened_weights[j])).item()

                eu_distances.append(eu_dist)
                cos_similarities.append(cos_sim)

        # Combine model update, cosine similarity, and Euclidean distance into cluster metrics for client i
        combined_metric = np.concatenate((flattened_weights[i], [np.mean(cos_similarities), np.mean(eu_distances)]))
        metrics.append(combined_metric)

    return np.array(metrics)


def compute_layerwise_trust_scores(features_list, tau=1):
    """
    Compute trust scores by calculating distances between multi-layer features.

    Args:
    - features_list: A list of feature sets (multi-layer features) for each client.
    - tau: Temperature parameter for softmax calculation of trust scores.

    Returns:
    - trust_scores: A list of trust scores for each client.
    """
    num_clients = len(features_list)
    layerwise_distances = np.zeros((num_clients, num_clients))  # Pairwise distance matrix

    # Calculate distances between features at each layer for every pair of clients
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            layer_distance = 0
            # Assuming each feature contains multi-layer features in sequential order
            for layer in range(len(features_list[i])):
                layer_i = features_list[i][layer]
                layer_j = features_list[j][layer]

                # Calculate Euclidean distance between corresponding layers
                dist = torch.dist(layer_i, layer_j).item()
                layer_distance += dist

            # Average layer distance across all layers
            avg_layer_distance = layer_distance / len(features_list[i])
            layerwise_distances[i, j] = avg_layer_distance
            layerwise_distances[j, i] = avg_layer_distance

    # Use inverse distances to compute trust scores
    trust_scores = []
    for i in range(num_clients):
        avg_distance = np.mean(layerwise_distances[i, :])  # Average distance of this client to all others
        inverse_distance = 1 / (avg_distance + 1e-6)  # Invert the distance to get a trustworthiness measure
        trust_scores.append(inverse_distance)

    # Apply softmax with temperature to normalize trust scores
    trust_scores = np.array(trust_scores)
    trust_scores = np.exp(trust_scores / tau) / np.sum(np.exp(trust_scores / tau))

    return trust_scores

def evaluate_tpr_npr(refined_clusters, local_malicious_indices):
    """
    Evaluate TPR, NPR, FPR, and Detection Rate.

    Args:
    - refined_clusters: Indices of the clients that are identified as benign after clustering.
    - local_malicious_indices: Known indices of malicious clients.

    Returns:
    - tpr: True Positive Rate.
    - npr: Negative Predictive Rate.
    - fpr: False Positive Rate.
    - detection_rate: Detection rate (same as TPR).
    """
    # True Positives (TP): Malicious clients correctly identified as malicious
    true_positives = len([idx for idx in local_malicious_indices if idx not in refined_clusters])

    # False Negatives (FN): Malicious clients missed by the detection (identified as benign)
    false_negatives = len([idx for idx in local_malicious_indices if idx in refined_clusters])

    # True Negatives (TN): Benign clients correctly identified as benign
    true_negatives = len([idx for idx in refined_clusters if idx not in local_malicious_indices])

    # False Positives (FP): Benign clients incorrectly flagged as malicious
    false_positives = len([idx for idx in refined_clusters if idx in local_malicious_indices])

    # True Positive Rate (TPR)
    tpr = true_positives / (true_positives + false_negatives + 1e-6)

    # Negative Predictive Rate (TNR)
    tnr = true_negatives / (true_negatives + false_positives + 1e-6)

    # False Positive Rate (FPR)
    fpr = false_positives / (false_positives + true_negatives + 1e-6)

    #
    # False Negative Rate (FNR)
    fnr = false_negatives / (false_negatives + true_positives + 1e-6)

    # Detection Rate (same as TPR)
    detection_rate = tpr

    return tpr, tnr, fpr, fnr, detection_rate

def save_detection_results_to_csv(iter_num, benign_indices, scores, malicious_indices, all_indices, all_norm_dist, tpr, tnr, fpr, fnr, detection_rate, timings, csvfile):
    """
    Save detection results along with MMD-based scores to a CSV file.

    :param iter_num: The iteration number.
    :param benign_indices: List of benign client indices.
    :param scores: MMD-based trust scores for each benign client.
    :param tpr: True Positive Rate.
    :param npr: Negative Predictive Rate.
    :param fpr: False Positive Rate.
    :param detection_rate: Detection rate of malicious clients.
    :param timings: Time taken for various steps.
    :param csvfile: The CSV file to save the results.
    """
    with open(csvfile, mode='a+', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file is empty
        if file.tell() == 0:
            writer.writerow(
                ['Iteration', 'Benign Indices', 'MMD Trust Scores', 'Malicious Indices','All Indices', 'All Norm Dist', 'TPR', 'TNR', 'FPR', 'FNR',
                 'Detection Rate', 'Timings'])

        # Write the row with detection results
        writer.writerow(
            [iter_num, benign_indices, scores, malicious_indices, all_indices, all_norm_dist, tpr, tnr, fpr, fnr, detection_rate, timings])

def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = update[key] - model[key]
    return update2

def Geminiguard(w_updates, w_locals_dict, net, central_dataset, dataset_test, global_parameters, iter,
                malicious_list, device, tau, attack_label, local_bs, plr_class):
    timings = []  # To record timings for each step

    # Step 1: Flatten model weights for each client
    start_time = time.time()
    print("\n[INFO] Step 1: Flattening model weights for clustering...\n")
    flattened_weights = [flatten_weights(w) for w in w_updates]
    print(f"[INFO] Flattened {len(flattened_weights)} model updates into vectors.\n")
    timings.append(time.time() - start_time)

    # Step 2: Compute cluster metrics (for each client's model update)
    start_time = time.time()
    print("[INFO] Step 2: Computing cluster metrics (model update + direction + magnitude)...\n")
    cluster_metrics = compute_cluster_metrics(flattened_weights)
    timings.append(time.time() - start_time)

    # Step 3: Perform K-means clustering using Silhouette Coefficient
    start_time = time.time()
    print("[INFO] Step 3: Clustering using K-means with dynamic cluster determination...\n")

    best_silhouette_score = -1
    best_k = 2  # Minimum number of clusters for K-means
    for k in range(2, 4):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_metrics)
        silhouette_avg = silhouette_score(cluster_metrics, cluster_labels)
        print(f"[INFO] Silhouette Score for k={k}: {silhouette_avg:.4f}")
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_k = k
    print(
        f"[INFO] Optimal number of clusters determined: k={best_k} with Silhouette Score={best_silhouette_score:.4f}\n")
    kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans_model.fit_predict(cluster_metrics)

    # Visualize clusters using PCA
    print("[INFO] Visualizing clusters with PCA...\n")
    pca = PCA(n_components=2)
    reduced_cluster_metrics = pca.fit_transform(cluster_metrics)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_cluster_metrics[:, 0], reduced_cluster_metrics[:, 1], c=clusters, cmap='coolwarm',
                marker='o')
    plt.title(f"K-means Clustering of Clients (Iteration {iter}) with k={best_k}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.savefig(f'./clusters3/clusters_plot_{iter}.png')

    # Identify benign clusters by filtering updates close to centroids within a threshold
    cluster_sizes = {cluster: len(np.where(clusters == cluster)[0]) for cluster in np.unique(clusters)}
    benign_indices = []
    all_indices = []
    all_norm_dis = []
    for cluster_label in cluster_sizes.keys():
        cluster_indices = np.where(clusters == cluster_label)[0]
        cluster_center = kmeans_model.cluster_centers_[cluster_label]
        # for idx in cluster_indices:
        #     if np.linalg.norm(cluster_metrics[idx] - cluster_center) <= args.tau:  # Threshold τ
        #         benign_indices.append(idx)

        # 存储当前聚类所有点的原始距离
        distances = [np.linalg.norm(cluster_metrics[idx] - cluster_center) for idx in cluster_indices]

        if distances:  # 确保距离列表非空
            max_dist = max(distances)
            min_dist = min(distances)

            if max_dist == min_dist:
                # 所有点距离相同，全部视为良性
                benign_indices.extend(cluster_indices)
                # continue
            else:
                for idx, dist in zip(cluster_indices, distances):
                    # 归一化到[0,1]范围
                    normalized_dist = (dist - min_dist) / (max_dist - min_dist)
                    all_indices.append(idx)
                    all_norm_dis.append(normalized_dist)
                    if normalized_dist <= tau:
                        benign_indices.append(idx)

    print(f"[INFO] Benign cluster selected with {len(benign_indices)} clients.\n")
    timings.append(time.time() - start_time)

    # Step 4: Multi-layer feature extraction for benign clients
    start_time = time.time()
    print("[INFO] Step 4: Extracting multi-layer features for benign clients...\n")
    w_feature = []
    for idx in benign_indices:
        local = LocalUpdate(dataset=dataset_test, idxs=central_dataset,  model=net, attack_label=attack_label, local_bs=local_bs, device=device)
        net.load_state_dict(w_locals_dict[idx])
        feature = local.get_multi_layers(net=copy.deepcopy(net).to(device), plr_class=plr_class)
        w_feature.append(feature)

    print(f"[INFO] Extracted multi-layer features for {len(w_feature)} clients.\n")
    timings.append(time.time() - start_time)

    # Step 5: MMD-based scoring (select top 50% nearest neighbors)
    start_time = time.time()
    print("[INFO] Step 5: Calculating MMD-based scores...\n")
    multi_layer_scores = compute_layerwise_trust_scores(w_feature, tau=1.0)
    print(f"[INFO] Calculated MMD-based scores for {len(multi_layer_scores)} clients.\n")
    for idx, score in enumerate(multi_layer_scores):
        print(f"[SCORE] Client {benign_indices[idx]}: MMD Trust Score = {score:.4f}")
    timings.append(time.time() - start_time)

    # Step 6: Aggregation based on trust scores
    start_time = time.time()
    print("\n[INFO] Step 6: Aggregating model updates based on MMD scores...\n")
    w_avg = copy.deepcopy(global_parameters)
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].float()
        for idx, benign_idx in enumerate(benign_indices):
            w_avg[k] += w_updates[benign_idx][k] * multi_layer_scores[idx]

    print("[COMPLETE] Model aggregation finished.\n")
    timings.append(time.time() - start_time)

    # Step 7: Evaluate TPR, NPR, and Detection Rate
    local_malicious_indices = [i for i in malicious_list]
    tpr, tnr, fpr, fnr, detection_rate = evaluate_tpr_npr(benign_indices, local_malicious_indices)
    print(
        f"[EVALUATION] TPR: {tpr:.4f}, TNR: {tnr:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f} Detection Rate: {detection_rate:.4f}")

    # Save results to CSV
    total_time = sum(timings)
    csvfile = f'detection_results.csv'
    save_detection_results_to_csv(iter, benign_indices, multi_layer_scores, malicious_list, all_indices,
                                  all_norm_dis, tpr, tnr, fpr, fnr, detection_rate, total_time, csvfile)

    return w_avg

def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)
#original fltrust
def fltrust(params, central_param, global_parameters, server_lr):
    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    sum_parameters = None
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        # 计算cos相似度得分和向量长度裁剪值
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm/torch.norm(local_parameters_v)
        score_list.append(client_cos)
        FLTrustTotalScore += client_cos
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                # 乘得分 再乘裁剪值
                sum_parameters[key] = client_cos * \
                    client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
                    var]
    if FLTrustTotalScore == 0:
        print(score_list)
        return global_parameters
    for var in global_parameters:
        # 除以所以客户端的信任得分总和
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * server_lr
    print(score_list)
    return global_parameters


def kernel_function(x, y):
    sigma = 1.0
    return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))

def compute_mmd(x, y):
    # Compute the MMD between two tensors x and y
    # x and y should have the same number of samples
    m = x.size(0)
    n = y.size(0)
    # Compute the kernel matrices for x and y
    xx_kernel = torch.zeros((m, m))
    yy_kernel = torch.zeros((n, n))
    xy_kernel = torch.zeros((m, n))
    for i in range(m):
        for j in range(i, m):
            xx_kernel[i, j] = xx_kernel[j, i] = kernel_function(x[i], x[j])

    for i in range(n):
        for j in range(i, n):
            yy_kernel[i, j] = yy_kernel[j, i] = kernel_function(y[i], y[j])

    for i in range(m):
        for j in range(n):
            xy_kernel[i, j] = kernel_function(x[i], y[j])
    # Compute the MMD statistic
    mmd = (torch.sum(xx_kernel) / (m * (m - 1))) + (torch.sum(yy_kernel) / (n * (n - 1))) - (2 * torch.sum(xy_kernel) / (m * n))
    return mmd

def flare(w_updates, w_locals, net, central_dataset, dataset_test, global_parameters, device, attack_label, local_bs):
    w_feature = []
    temp_model = copy.deepcopy(net)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()

    for client in w_locals:
        net.load_state_dict(client)
        # local = LocalUpdate(args=args, dataset=dataset_test, idxs=central_dataset)
        local = LocalUpdate(dataset=dataset_test, idxs=central_dataset, model=net, attack_label=attack_label,
                            local_bs=local_bs, device=device)
        feature = local.get_PLR(net=copy.deepcopy(net).to(device))
        w_feature.append(feature)
    distance_list = [[] for i in range(len(w_updates))]
    # distance_list=[list(len(w_updates)) for i in range(len(w_updates))]
    for i in range(len(w_updates)):
        for j in range(i + 1, len(w_updates)):
            score = compute_mmd(w_feature[i], w_feature[j])
            distance_list[i].append(score.item())
            distance_list[j].append(score.item())
    print('defense line121 distance_list', distance_list)
    vote_counter = [0 for i in range(len(w_updates))]
    k = round(len(w_updates) * 0.5)
    for i in range(len(w_updates)):
        IDs = np.argsort(distance_list[i])
        for j in range(len(IDs)):
            # client_id is the index of client i-th client voting for
            # distance_list[] only records score with other clients without itself
            # so distance_list[i][i] should be itself
            # client_id = j + 1 after j >= i
            if IDs[j] >= i:
                client_id = IDs[j] + 1
            else:
                client_id = IDs[j]
            vote_counter[client_id] += 1
            if j + 1 >= k:  # first 𝑘 elements in 𝐼 𝐷𝑠 and vote for it
                break

    trust_score = [x / sum(vote_counter) for x in vote_counter]
    # print('defense line188 len trust_score', trust_score)

    w_avg = copy.deepcopy(global_parameters)
    for k in w_avg.keys():
        for i in range(0, len(w_updates)):
            try:
                w_avg[k] += w_updates[i][k] * trust_score[i]
            except:
                print("Fed.py line17 type_as", 'w_updates[i][k].type():', w_updates[i][k].type(), k)
                w_updates[i][k] = w_updates[i][k].type_as(w_avg[k]).long()
                w_avg[k] = w_avg[k].long() + w_updates[i][k] * trust_score[i]
    return w_avg

# -------------------------
# Evaluation helper
# -------------------------
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total
# -------------------------
# Average weights helper
# -------------------------
def average_weights(local_weights):
    """Compute element-wise average of a list of state_dicts."""
    avg_w = {}
    for k in local_weights[0].keys():
        avg_w[k] = torch.mean(torch.stack([w[k].float() for w in local_weights]), dim=0)
    return avg_w
# -------------------------
# FLShield defense
# -------------------------
def flshield_defense(local_weights, global_model, val_dataset, num_clusters=3, bijective=False):
    """
    FLShield-style defense:
    - Cluster client updates.
    - Select best cluster by validation accuracy.
    - Return aggregated state_dict.
    - Supports bijective FLShield if bijective=True.
    """
    # Flatten client updates
    updates = np.array([flatten_weights(w) for w in local_weights])

    # Optionally apply bijective transformation (placeholder)
    if bijective:
        # Example: simple normalization or other bijective mapping
        updates = (updates - updates.mean(axis=0)) / (updates.std(axis=0) + 1e-8)

    # KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(updates)
    labels = kmeans.labels_

    # Evaluate cluster representatives
    cluster_scores = []
    for c in range(num_clusters):
        members = [local_weights[i] for i in range(len(local_weights)) if labels[i] == c]
        if len(members) == 0:
            cluster_scores.append(-1)
            continue
        agg_w = average_weights(members)
        global_model.load_state_dict(agg_w)
        acc = evaluate(global_model, DataLoader(val_dataset, batch_size=256),
                       device=next(global_model.parameters()).device)
        cluster_scores.append(acc)

    best_cluster = np.argmax(cluster_scores)
    chosen = [local_weights[i] for i in range(len(local_weights)) if labels[i] == best_cluster]

    return average_weights(chosen)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)
def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters

def flame(local_model, update_params, global_model, wrong_mal, right_ben, turn, noise, debug=False):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list = []
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    if debug == True:
        filename = './' + 'flame' + '/flame_analysis.txt'
        f = open(filename, "a")
        for i in cos_list:
            f.write(str(i))
            print(i)
            f.write('\n')
        f.write('\n')
        f.write("--------Round--------")
        f.write('\n')
    num_clients = max(int(0.1 * 100), 1)
    num_malicious_clients = int(0 * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(
        cos_list)
    print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster = 0
    max_cluster_index = 0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list, torch.norm(parameters_dict_to_vector(update_params[i]), p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max() + 1):
            if len(clusterer.labels_[clusterer.labels_ == index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_ == index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
                norm_list = np.append(norm_list, torch.norm(parameters_dict_to_vector(update_params[i]),
                                                            p=2).item())  # no consider BN
    print(benign_client)

    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            wrong_mal += 1
        else:
            #  minus per benign in cluster
           right_ben += 1
    turn += 1

    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value / norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
    # add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0, std=noise * clip_value)
        var += temp
    return global_model

class Server:
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, fo, current_time, malnum, malclient,
                 poisonratio, poison_label, attack_method, per_epoch, defense, central_dataset, dataset_test, tau, local_bs, plr_class,
                 server_lr, wrong_mal, right_ben, turn, noise):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.defense = defense
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)  # 复制模型
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_global_train_acc, self.rs_global_train_loss, self.rs_global_test_acc, self.rs_local_train_acc_per, self.rs_local_train_loss_per, self.rs_local_test_acc_per = [], [], [], [], [], []
        self.rs_global_train_asr, self.rs_global_train_asr_loss, self.rs_global_test_asr, self.rs_local_train_asr_per, self.rs_local_train_asr_loss_per, self.rs_local_test_asr_per = [], [], [], [], [], []
        self.times = times
        self.fo = fo
        self.current_time = current_time
        self.malnum = malnum
        self.malclient = malclient
        self.poisonratio = poisonratio
        self.poisonlabel = poison_label
        self.attack_method = attack_method
        self.folder_path = f'results/{self.dataset}_{current_time}_{algorithm}_{attack_method}_{defense}_{poisonratio}_{per_epoch}'
        self.mi_path = f'results/{self.dataset}_{current_time}_{algorithm}_{attack_method}_{poisonratio}_{per_epoch}/'
        self.savedmodelpath = f'saved_model'

        self.central_dataset = central_dataset
        self.dataset_test = dataset_test
        self.tau = tau
        self.local_bs = local_bs
        self.plr_class = plr_class

        self.server_lr=server_lr
        self.wrong_mal = wrong_mal
        self.right_ben = right_ben
        self.turn = turn
        self.noise = noise
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            print('Folder already exists')

    def aggregate_grads(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        for user in self.users:
            user.set_parameters(self.model)

    def send_pmodel_parma(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        for user in self.users:
            user.set_parameters(user.pmodel)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            newvalue = server_param.data + user_param.data.clone() * ratio
            server_param.data.copy_(newvalue)

    def model_dist_norm(self, user):
        squared_sum = 0
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            squared_sum += torch.sum(torch.pow(server_param.data.clone() - user_param.data.clone(), 2))

        return math.sqrt(squared_sum)

    def aggregate_parameters(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"

        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0

        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            # self.add_parameters(user, 1/ len(self.selected_users))

    def Trimmed_Mean(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"

        clients_params = []
        for user in self.selected_users:
            clients_params.append(
                np.concatenate([param.data[:].cpu().numpy().flatten() for param in user.get_parameters()]))
        clients_params = torch.tensor(np.array(clients_params))

        m = 0
        for user in self.selected_users:
            if user.id in self.malclient:
                m += 1

        a = clients_params.sort(dim=0)[0][m:len(self.selected_users) - m]
        b = torch.mean(a, dim=0)

        # 全局模型归零
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        # 模型参数更新
        offset = 0
        for server_param in self.model.parameters():
            with torch.no_grad():
                new_size = functools.reduce(lambda x, y: x * y, server_param.shape)
                new_value = b[offset:offset + new_size]
                server_param.data[:] = new_value.clone().detach().reshape(server_param.shape)
                offset += new_size

    def Multi_Krum(self):
        def krum_create_distances(clients_params):
            distances = defaultdict(dict)
            for i in range(len(clients_params)):
                for j in range(i):
                    distances[i][j] = distances[j][i] = np.linalg.norm(clients_params[i] - clients_params[j])
            return distances

        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        clients_params = []
        for user in self.selected_users:
            clients_params.append(
                np.concatenate([param.data[:].cpu().numpy().flatten() for param in user.get_parameters()]))
        clients_params = np.array(clients_params)

        m = 0
        for user in self.selected_users:
            if user.id in self.malclient:
                m += 1
        non_malicious_count = len(self.selected_users) - m

        distances = krum_create_distances(clients_params)

        selection_set = []
        krumscore = []

        for user in distances.keys():
            errors = sorted(distances[user].values())
            current_error = sum(errors[:non_malicious_count])
            krumscore.append(current_error)
        tmp = krumscore[0]
        krumscore[0] = krumscore[1]
        krumscore[1] = tmp
        result = map(krumscore.index, heapq.nsmallest(non_malicious_count, krumscore))
        for i in result:
            selection_set.append(clients_params[i])

        result_params = torch.tensor(selection_set).mean(dim=0)

        # 全局模型归零
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        # 模型参数更新
        offset = 0
        for server_param in self.model.parameters():
            with torch.no_grad():
                new_size = functools.reduce(lambda x, y: x * y, server_param.shape)
                new_value = result_params[offset:offset + new_size]
                server_param.data[:] = new_value.clone().detach().reshape(server_param.shape)
                offset += new_size

    def Geminiguard_Aggregate(self, glob_iter):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        w_glob = self.model.state_dict()
        w_updates = []
        w_client = []

        for i, user in enumerate(self.selected_users):
            w = user.model.state_dict()
            # w_client.append(copy.deepcopy(user.model).to("cuda:0"))
            w_client.append(w)
            w_updates.append(get_update(w, w_glob))
        print("Now we use another dataset for validation")
        malclient_positions = []
        for mal in self.malclient:
            # 查找第一个匹配的对象索引
            for idx, obj in enumerate(self.selected_users):
                if obj.id == mal:
                    malclient_positions.append(idx)
                    break  # 找到第一个就停止
        w_avg =Geminiguard(w_updates=w_updates, w_locals_dict = w_client, net=self.model, central_dataset=self.central_dataset, dataset_test=self.dataset_test,  global_parameters=w_glob, iter= glob_iter,
                malicious_list=malclient_positions, device=self.device, tau = self.tau, attack_label=self.poisonlabel, local_bs=self.local_bs, plr_class=self.plr_class)
        self.model.load_state_dict(w_avg)

    def Fltrust_Aggregate(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        w_glob = self.model.state_dict()
        w_updates = []
        for i, user in enumerate(self.selected_users):
            w = user.model.state_dict()
            w_updates.append(get_update(w, w_glob))

        local = LocalUpdate(dataset=self.dataset_test, idxs=self.central_dataset,  model=self.model, attack_label=self.poisonlabel, local_bs=self.local_bs, device=self.device)

        fltrust_norm, loss = local.train(
            net=copy.deepcopy(self.model).to(self.device))
        fltrust_norm = get_update(fltrust_norm, w_glob)
        # print(idxs_users)
        # def fltrust(params, central_param, global_parameters, args):
        w_glob = fltrust(w_updates, fltrust_norm, w_glob, self.server_lr)
        self.model.load_state_dict(w_glob)

    def Flare_Aggregate(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        w_glob = self.model.state_dict()
        w_updates = []
        w_client = []

        for i, user in enumerate(self.selected_users):
            w = user.model.state_dict()
            # w_client.append(copy.deepcopy(user.model).to("cuda:0"))
            w_client.append(w)
            w_updates.append(get_update(w, w_glob))

        w_glob = flare(w_updates=w_updates, w_locals=w_client, net=self.model, central_dataset=self.central_dataset, dataset_test=self.dataset_test, global_parameters=w_glob, device=self.device, attack_label=self.poisonlabel, local_bs=self.local_bs)
        self.model.load_state_dict(w_glob)

    def Flshield_Aggregate(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        w_glob = self.model.state_dict()
        w_updates = []
        w_client = []

        for i, user in enumerate(self.selected_users):
            w = user.model.state_dict()
            # w_client.append(copy.deepcopy(user.model).to("cuda:0"))
            w_client.append(w)
            w_updates.append(get_update(w, w_glob))

        # w_glob = flare(w_updates=w_updates, w_locals=w_client, net=self.model, central_dataset=self.central_dataset, dataset_test=self.dataset_test, global_parameters=w_glob, device=self.device, attack_label=self.poisonlabel, local_bs=self.local_bs)
        w_glob = flshield_defense(local_weights=w_client, global_model=self.model, val_dataset=self.dataset_test)
        self.model.load_state_dict(w_glob)

    def Flame_Aggregate(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        w_glob = self.model.state_dict()
        w_updates = []
        w_client = []

        for i, user in enumerate(self.selected_users):
            w = user.model.state_dict()
            # w_client.append(copy.deepcopy(user.model).to("cuda:0"))
            w_client.append(w)
            w_updates.append(get_update(w, w_glob))

        # w_glob = flame(w_updates=w_updates, w_locals=w_client, net=self.model, central_dataset=self.central_dataset, dataset_test=self.dataset_test, global_parameters=w_glob, device=self.device, attack_label=self.poisonlabel, local_bs=self.local_bs)
        w_glob = flame(local_model=w_client, update_params=w_updates, global_model=w_glob, wrong_mal=self.wrong_mal,
                       right_ben=self.right_ben, turn=self.turn, noise=self.noise, debug=False)
        self.model.load_state_dict(w_glob)

    def save_model(self):
        model_path = os.path.join(f'{self.folder_path}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def save_model_epoch(self, global_iteration):
        model_path = os.path.join(f'{self.folder_path}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, str(global_iteration) + "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join(f'{self.savedmodelpath}', "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def load_model_pretrain(self, method):
        model_path = os.path.join(f'{self.savedmodelpath}', str(method) + "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join(f'{self.folder_path}', self.dataset, "server" + ".pt"))

    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if (num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        return np.random.choice(self.users, num_users, replace=False)

    def select_users_fixed(self, round_idx, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        '''
        if (num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))

        # 分离恶意和良性客户端
        malicious_users = [user for user in self.users if user.id in self.malclient]
        benign_users = [user for user in self.users if user.id not in self.malclient]

        # 计算应该选择的恶意客户端数量（按比例）
        total_malicious = len(malicious_users)
        total_benign = len(benign_users)

        # 计算应该选择的恶意客户端数量（保持与总体相同的比例）
        num_malicious = max(1, round(num_users * total_malicious / len(self.users)))
        num_benign = num_users - num_malicious

        # 确保不会选择超过可用数量的客户端
        num_malicious = min(num_malicious, total_malicious)
        num_benign = min(num_benign, total_benign)

        # 如果数量不足，调整另一组的数量
        if num_malicious + num_benign < num_users:
            if num_malicious < total_malicious:
                num_malicious = min(total_malicious, num_malicious + (num_users - num_malicious - num_benign))
            else:
                num_benign = min(total_benign, num_benign + (num_users - num_malicious - num_benign))

        # 随机选择恶意和良性客户端
        selected_malicious = np.random.choice(malicious_users, num_malicious,
                                              replace=False) if num_malicious > 0 else []
        selected_benign = np.random.choice(benign_users, num_benign, replace=False) if num_benign > 0 else []

        # 合并结果
        selected_users = np.concatenate([selected_malicious, selected_benign])

        # 打乱顺序以避免模式识别
        np.random.shuffle(selected_users)

        return selected_users

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self, user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def persionalized_aggregate_parameters(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        # if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            # self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    def persionalized_Multi_Krum(self):
        previous_param = copy.deepcopy(list(self.model.parameters()))

        self.Multi_Krum()

        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    def persionalized_Trimmed_Mean(self):
        previous_param = copy.deepcopy(list(self.model.parameters()))

        self.Trimmed_Mean()

        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if (self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_global_test_acc) != 0 & len(self.rs_global_train_acc) & len(self.rs_global_train_loss)):
            with h5py.File('{}'.format(self.folder_path) + '/' + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_global_test_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_global_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_global_train_loss)
                hf.close()

        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if (self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_local_test_acc_per) != 0 & len(self.rs_local_train_acc_per) & len(
                self.rs_local_train_loss_per)):
            with h5py.File('{}'.format(self.folder_path) + '/' + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_local_test_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_local_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_local_train_loss_per)
                hf.close()

    def save_poison_results(self):
        alg = self.dataset + "_" + self.algorithm + "_poison"
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)

        if (self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)

        alg = alg + "_" + str(self.times)

        if (len(self.rs_global_test_asr) != 0 & len(self.rs_global_train_asr) & len(self.rs_global_train_asr_loss)):
            with h5py.File('{}'.format(self.folder_path) + '/' + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_asr', data=self.rs_global_test_asr)
                hf.create_dataset('rs_train_asr', data=self.rs_global_train_asr)
                hf.create_dataset('rs_train_asr_loss', data=self.rs_global_train_asr_loss)
                hf.close()

        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p" + "_poison"
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)

        if (self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)

        alg = alg + "_" + str(self.times)

        if (len(self.rs_local_test_asr_per) != 0 & len(self.rs_local_train_asr_per) & len(
                self.rs_local_train_asr_loss_per)):
            with h5py.File('{}'.format(self.folder_path) + '/' + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_asr', data=self.rs_local_test_asr_per)
                hf.create_dataset('rs_train_asr', data=self.rs_local_train_asr_per)
                hf.create_dataset('rs_train_asr_loss', data=self.rs_local_train_asr_loss_per)

    def test(self):
        '''tests self.latest_model on given clients
        在所有用户上用自己的数据基于自己的本地模型测试
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def poison_test(self, poiosnlabel, trigger, pattern):
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.poisontest(poiosnlabel, trigger, pattern)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        # 所有用户在训练数据集上的acc、loss、id、样本数，基于自己模型
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def poison_train_error_and_loss(self, poiosnlabel, trigger, pattern):
        # 所有用户在训练数据集上的acc、loss、id、样本数，基于自己模型
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.poison_train_error_and_loss(poiosnlabel=poiosnlabel, trigger=trigger, pattern=pattern)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()  # 每个用户的基于全局模型、测试数据的准确率，样本数、id
        stats_train = self.train_error_and_loss()  # 每个用户的基于全局模型、训练数据的准确率，样本数、id、loss
        global_test_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        global_train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        global_train_loss = 0

        print("Average Global Accurancy: ", global_test_acc)
        print("Average Global Trainning Accurancy: ", global_train_acc)
        print()

        mal_person_acc = []
        ben_person_acc = []
        for i in range(len(stats[0])):
            if stats[0][i] in self.malclient:
                mal_person_acc.append(stats[2][i] * 1.0 / stats[1][i])
            else:
                ben_person_acc.append(stats[2][i] * 1.0 / stats[1][i])

        print("global benign client acc list:{}; mean acc:{}".format(ben_person_acc,
                                                                     sum(ben_person_acc) / len(ben_person_acc)))
        print("global malicious client acc list:{}; mean acc:{}".format(mal_person_acc,
                                                                        sum(mal_person_acc) / len(mal_person_acc)))

        return global_test_acc, global_train_acc, global_train_loss, sum(ben_person_acc) / len(ben_person_acc), sum(
            mal_person_acc) / len(mal_person_acc)

    def poison_evaluate(self, trigger, pattern):
        # 每个用户的基于全局模型、测试数据的准确率，样本数、id
        stats = self.poison_test(trigger=trigger, poiosnlabel=self.poisonlabel, pattern=pattern)
        # 每个用户的基于全局模型、训练数据的准确率，样本数、id、loss
        stats_train = self.poison_train_error_and_loss(trigger=trigger, poiosnlabel=self.poisonlabel, pattern=pattern)

        global_test_asr = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        global_train_asr = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        global_train_asr_loss = 0

        print("Average Global ATTACK ALL ASR: ", global_test_asr)
        print("Average Global ATTACK ALL train ASR: ", global_train_asr)

        mal_person_asr = []
        ben_person_asr = []
        for i in range(len(stats[0])):
            if stats[0][i] in self.malclient:
                mal_person_asr.append(stats[2][i] * 1.0 / stats[1][i])
            else:
                if stats[1][i] != 0:
                    ben_person_asr.append(stats[2][i] * 1.0 / stats[1][i])

        print("global benign client asr list:{}; mean asr:{}".format(ben_person_asr,
                                                                     sum(ben_person_asr) / len(ben_person_asr)))
        print("global malicious client asr list:{}; mean asr:{}".format(mal_person_asr,
                                                                        sum(mal_person_asr) / len(mal_person_asr)))

        return global_test_asr, global_train_asr, global_train_asr_loss, sum(ben_person_asr) / len(ben_person_asr), \
               sum(mal_person_asr) / len(mal_person_asr)

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        # print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        # print("Average Personal Trainning Loss: ", train_loss)

    def evaluate_one_step(self, per_epoch, trigger, pattern):
        for c in self.users:
            c.train_one_step(per_epoch)  # 每个用户训练一次，表示本地 fine-tune

        stats = self.test()
        stats_train = self.train_error_and_loss()

        # trigger is a list
        poison_stats = self.poison_test(trigger=trigger, poiosnlabel=self.poisonlabel, pattern=pattern)
        poison_stats_train = self.poison_train_error_and_loss(trigger=trigger, poiosnlabel=self.poisonlabel,
                                                              pattern=pattern)

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)  # 本地 fine-tune 结束后，恢复之前的模型，重新训练

        per_local_test_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        per_localtrain_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        per_localtrain_loss = 0
        print("Average Personal Accurancy (k local SGD): ", per_local_test_acc)
        print("Average Personal Trainning Accurancy (k local SGD): ", per_localtrain_acc)

        mal_person_acc = []
        ben_person_acc = []
        for i in range(len(stats[0])):
            if stats[0][i] in self.malclient:
                mal_person_acc.append(stats[2][i] * 1.0 / stats[1][i])
            else:
                ben_person_acc.append(stats[2][i] * 1.0 / stats[1][i])

        print("global benign client acc list:{}; mean acc:{}".format(ben_person_acc,
                                                                     sum(ben_person_acc) / len(ben_person_acc)))
        print("global malicious client acc list:{}; mean acc:{}".format(mal_person_acc,
                                                                        sum(mal_person_acc) / len(mal_person_acc)))

        per_local_test_asr = np.sum(poison_stats[2]) * 1.0 / np.sum(poison_stats[1])
        per_localtrain_asr = np.sum(poison_stats_train[2]) * 1.0 / np.sum(poison_stats_train[1])
        per_localtrain_losssr = 0
        print("Average Personal ATTACK ALL ASR (k local SGD): ", per_local_test_asr)
        print("Average Personal ATTACK ALL Trainning ASR (k local SGD): ", per_localtrain_asr)

        mal_person_asr = []
        ben_person_asr = []
        for i in range(len(poison_stats[0])):
            if poison_stats[0][i] in self.malclient:
                mal_person_asr.append(poison_stats[2][i] * 1.0 / poison_stats[1][i])
            else:
                if poison_stats[1][i] != 0:
                    ben_person_asr.append(poison_stats[2][i] * 1.0 / poison_stats[1][i])

        print("person benign client asr list:{}; mean asr:{}".format(ben_person_asr,
                                                                     sum(ben_person_asr) / len(ben_person_asr)))
        print("person malicious client asr list:{}; mean asr:{}".format(mal_person_asr,
                                                                        sum(mal_person_asr) / len(mal_person_asr)))

        return per_local_test_acc, per_localtrain_acc, per_localtrain_loss, per_local_test_asr, per_localtrain_asr, per_localtrain_losssr, \
               sum(ben_person_asr) / len(ben_person_asr), sum(mal_person_asr) / len(mal_person_asr), sum(
            ben_person_acc) / len(ben_person_acc), sum(mal_person_acc) / len(mal_person_acc)

    def evaluate_one_step_poison(self, per_epoch, trigger, pattern):
        for c in self.users:
            if c.id in self.malclient:
                c.train_one_step_poison(per_epoch, trigger=trigger, pattern=pattern, poison_label=self.poisonlabel,
                                        poison_ratio=self.poisonratio)
            else:
                c.train_one_step(per_epoch)  # 每个用户训练一次，表示本地 fine-tune

        stats = self.test()
        stats_train = self.train_error_and_loss()

        # trigger is a list
        poison_stats = self.poison_test(trigger=trigger, poiosnlabel=self.poisonlabel, pattern=pattern)
        poison_stats_train = self.poison_train_error_and_loss(trigger=trigger, poiosnlabel=self.poisonlabel,
                                                              pattern=pattern)

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)  # 本地 fine-tune 结束后，恢复之前的模型，重新训练

        per_local_test_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        per_localtrain_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        per_localtrain_loss = 0
        print("Average Personal Accurancy (k local SGD): ", per_local_test_acc)
        print("Average Personal Trainning Accurancy (k local SGD): ", per_localtrain_acc)
        mal_person_acc = []
        ben_person_acc = []
        for i in range(len(stats[0])):
            if stats[0][i] in self.malclient:
                mal_person_acc.append(stats[2][i] * 1.0 / stats[1][i])
            else:
                ben_person_acc.append(stats[2][i] * 1.0 / stats[1][i])

        print("global benign client acc list:{}; mean acc:{}".format(ben_person_acc,
                                                                     sum(ben_person_acc) / len(ben_person_acc)))
        print("global malicious client acc list:{}; mean acc:{}".format(mal_person_acc,
                                                                        sum(mal_person_acc) / len(mal_person_acc)))

        per_local_test_asr = np.sum(poison_stats[2]) * 1.0 / np.sum(poison_stats[1])
        per_localtrain_asr = np.sum(poison_stats_train[2]) * 1.0 / np.sum(poison_stats_train[1])
        per_localtrain_losssr = 0
        print("Average Personal ATTACK ALL ASR (k local SGD): ", per_local_test_asr)
        print("Average Personal ATTACK ALL Trainning ASR (k local SGD): ", per_localtrain_asr)

        mal_person_asr = []
        ben_person_asr = []
        for i in range(len(poison_stats[0])):
            if poison_stats[0][i] in self.malclient:
                mal_person_asr.append(poison_stats[2][i] * 1.0 / poison_stats[1][i])
            else:
                if poison_stats[1][i] != 0:
                    ben_person_asr.append(poison_stats[2][i] * 1.0 / poison_stats[1][i])

        print("person benign client asr list:{}; mean asr:{}".format(ben_person_asr,
                                                                     sum(ben_person_asr) / len(ben_person_asr)))
        print("person malicious client asr list:{}; mean asr:{}".format(mal_person_asr,
                                                                        sum(mal_person_asr) / len(mal_person_asr)))

        return per_local_test_acc, per_localtrain_acc, per_localtrain_loss, per_local_test_asr, per_localtrain_asr, per_localtrain_losssr, \
               sum(ben_person_asr) / len(ben_person_asr), sum(mal_person_asr) / len(mal_person_asr), sum(
            ben_person_acc) / len(ben_person_acc), sum(
            mal_person_acc) / len(mal_person_acc)

    def trigger_evasion_mnist(self, model, trigger, glob_iter, attackstart):
        models = copy.deepcopy(model)
        models.eval()
        init = False
        pre_trigger = torch.tensor(trigger[0]).cuda()
        new_trigger_list = []

        dataset = []

        for labelindex in range(10):
            count = 1
            for user in self.users:
                if user.id in self.malclient:
                    for X, Y in user.trainloaderfull:
                        for i in range(len(X)):
                            if Y[i] == labelindex and count < 100:
                                dataset.append((X[i], Y[i]))
                                count += 1
                            if count >= 100:
                                break
                        if count >= 100:
                            break
                if count >= 100:
                    break

        dataloaders = DataLoader(dataset, batch_size=64, shuffle=True)

        iter_dataloader = iter(dataloaders)

        def get_batch():
            try:  # Samples a new batch for persionalizing
                (X, y) = next(iter_dataloader)
            except StopIteration:
                iter_trainloader = iter(dataloaders)
                (X, y) = next(iter_trainloader)
            return (X.to(self.device), y.to(self.device))

        for e in range(1, 51, 1):  # learning loss 51,  17
            corrects = 0
            datas, labels = get_batch()  # 取出一个batch数据
            x = Variable(datas)
            y = Variable(labels)
            y_target = torch.LongTensor(y.size()).fill_(1)
            y_target = Variable(y_target, requires_grad=False).to(self.device)
            if not init:
                noise = copy.deepcopy(pre_trigger)
                noise = Variable(noise, requires_grad=True).to(self.device)
                init = True

            #image对应位置先置0，再加 noise
            for index in range(0, len(x)):
                for i in range(10, 20, 1):
                    for j in range(10, 20, 1):
                        x[index][0][i][j] = 0

            output = model((x + noise).float())
            classloss = nn.functional.cross_entropy(output, y_target)

            loss = classloss
            model.zero_grad()
            if noise.grad:
                noise.grad.fill_(0)
            loss.backward(retain_graph=True)

            noise = noise - noise.grad * 0.1

            for i in range(28):
                for j in range(28):
                    if i in range(10, 20, 1) and j in range(10, 20, 1):
                        continue
                    else:
                        noise[0][i][j] = 0

            noise = torch.clamp(noise, -1, 1)
            noise = Variable(noise, requires_grad=True).to(self.device)
            pred = output.data.max(1)[1]
            correct = torch.eq(pred, y_target).float().mean().item()
            corrects += pred.eq(y_target.data.view_as(pred)).cpu().sum().item()

            print('batchid:{},correct:{},noise:{}'.format(e, correct * 100, noise.data.norm()))

        for i in range(10):
            new_trigger_list.append(copy.deepcopy(noise))

        return new_trigger_list

    def trigger_all_mnist_l2(self, trigger, pattern, attackstart, intinal_trigger, glob_iter):
        models = copy.deepcopy(self.model)
        models.eval()
        net_parameters = list(models.parameters())
        noise = copy.deepcopy(trigger[0]).to(self.device)
        noise = Variable(noise, requires_grad=True)
        new_trigger_list = []

        dataset = []
        for labelindex in range(10):
            count = 1
            for user in self.users:
                if user.id in self.malclient:
                    for X, Y in user.trainloaderfull:
                        for i in range(len(X)):
                            if Y[i] == labelindex and count < 100:
                                dataset.append((X[i], Y[i]))
                                count += 1
                            if count >= 100:
                                break
                        if count >= 100:
                            break
                if count >= 100:
                    break

        dataloaders = DataLoader(dataset, batch_size=64, shuffle=True)

        noisetemp = torch.zeros((1, 28, 28)).float().to(self.device)

        for i in range(0, len(pattern)):
            pos = pattern[i]
            noisetemp[0][pos[0]][pos[1]] = 1

        round = 30 + 1
        if self.algorithm == "PerAvg-HF":
            round = 10 + 1

        for e in range(1, round, 1):
            total_loss = 0
            for batch_id, (datas, labels) in enumerate(dataloaders):
                x = Variable(datas.to(self.device))
                y = Variable(labels.to(self.device))

                y_target = torch.LongTensor(y.size()).fill_(int(self.poisonlabel))
                y_target = Variable(y_target.to(self.device), requires_grad=False)

                for param in list(models.parameters()):
                    param.requires_grad = True

                output_nor = models((x).float())
                loss_nor = nn.functional.cross_entropy(output_nor, y)
                grad_nor = torch.autograd.grad(loss_nor, net_parameters)
                grad_nor = list((_.detach().clone() for _ in grad_nor))

                #image 对应的pattern位置先置 0， 后加 noise
                patterntensor = torch.ones((1, 28, 28)).float().to(self.device)
                for i in range(0, len(pattern)):
                    pos = pattern[i]
                    patterntensor[0][pos[0]][pos[1]] = 0

                patterntensor = patterntensor.unsqueeze(0)
                patterntensor = patterntensor.repeat(len(datas), 1, 1, 1)

                x = x * patterntensor

                output_mal = models((x + noise).float())
                loss_mal = nn.functional.cross_entropy(output_mal, y_target)
                grad_mal = torch.autograd.grad(loss_mal, net_parameters, create_graph=True)

                #gradient matching
                loss = self.match_l2_loss(grad_mal, grad_nor)
                total_loss += loss.item()

                models.zero_grad()
                if noise.grad:
                    noise.grad.fill_(0)

                loss.backward(retain_graph=True)
                noise = noise - noise.grad * 0.1
                noise = noise * noisetemp
                noise = torch.clamp(noise, -1, 1)

                noise = Variable(noise.data, requires_grad=True)

            print('l2 loss:{}'.format(total_loss))

        for i in range(10):
            new_trigger_list.append(copy.deepcopy(noise))

        return new_trigger_list


    def proj_lp(self, v, xi, p):
        # Project on the lp ball centered at 0 and of radius xi
        # SUPPORTS only p = 2 and p = Inf for now
        if p == 2:
            v = v * min(1, xi / torch.norm(v))
            # v = v / np.linalg.norm(v.flatten(1)) * xi
        elif p == np.inf:
            v = np.sign(v) * np.minimum(abs(v), xi)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported...')
        return v

    def match_loss(self, grad_mal, grad_nor):
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(grad_nor)):
            gw_real_vec.append(grad_nor[ig].reshape((-1)))
            gw_syn_vec.append(grad_mal[ig].reshape((-1)))

        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

        return dis

    def match_l2_loss(self, grad_mal, grad_nor):
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(grad_nor)):
            gw_real_vec.append(grad_nor[ig].reshape((-1)))
            gw_syn_vec.append(grad_mal[ig].reshape((-1)))

        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)

        dis = torch.sqrt(torch.sum((gw_syn_vec - gw_real_vec) ** 2))

        return dis

    def mi_loss(self, grad_mal, grad_nor, mi_model):
        mi_model.eval()
        gw_nor_vec = []
        gw_mal_vec = []
        for ig in range(len(grad_mal)):
            gw_nor_vec.append(grad_mal[ig].reshape((-1)))
            gw_mal_vec.append(grad_nor[ig].reshape((-1)))

        gw_nor_vec = torch.cat(gw_nor_vec, dim=0)
        gw_mal_vec = torch.cat(gw_mal_vec, dim=0)
        mi_lb, t, et = self.mutual_information(gw_nor_vec, gw_mal_vec, mi_model)

        return mi_lb

    def learn_mine(self, x, y, mine_net, mine_net_optim, ma_et, ma_rate=0.0001):
        # 梯度进行处理
        gw_nor_vec = []
        gw_mal_vec = []
        for ig in range(len(x)):
            gw_nor_vec.append(x[ig].reshape((-1)))
            gw_mal_vec.append(y[ig].reshape((-1)))

        gw_nor_vec = torch.cat(gw_nor_vec, dim=0)

        gw_mal_vec = torch.cat(gw_mal_vec, dim=0)

        mi_lb, t, et = self.mutual_information(gw_nor_vec, gw_mal_vec, mine_net)
        ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

        # unbiasing use moving average
        # loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
        # use biased estimator
        loss = - mi_lb

        mine_net_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mine_net.parameters(), max_norm=20, norm_type=2)
        mine_net_optim.step()
        return mi_lb, ma_et

    def mutual_information(self, joint, marginal, mine_net):
        t = mine_net(joint)
        et = torch.exp(mine_net(marginal))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et

    def train_mine_estimator(self, grad_nor, grad_mal, mine_net, mine_net_optim, batch_size=100, iter_num=int(5e+3),
                             log_freq=int(1e+3)):
        # data is grad_nor and grad_mal
        result = list()
        ma_et = 1.
        for i in range(iter_num):
            mi_lb, ma_et = self.learn_mine(grad_nor, grad_mal, mine_net, mine_net_optim, ma_et)
            result.append(mi_lb.detach().cpu().numpy())
            if (i + 1) % (log_freq) == 0:
                print(result[-1])
        return result

    def compute_grad_mask(self, models, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model = copy.deepcopy(models)
        model.train()
        model.zero_grad()

        dataset = []
        for user in self.users:
            if user.id in self.malclient:
                for X, Y in user.trainloaderfull:
                    for i in range(len(X)):
                        dataset.append((X[i], Y[i]))
        dataloaders = DataLoader(dataset, self.batch_size)

        for batch_id, (datas, labels) in enumerate(dataloaders):
            input = datas.to(self.device)
            label = labels.to(self.device)
            output = model(input)
            loss = nn.functional.cross_entropy(output, label)
            loss.backward(retain_graph=True)

        mask_grad_list = []
        grad_list = []
        grad_abs_sum_list = []
        k_layer = 0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                grad_list.append(parms.grad.abs().view(-1))
                grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())
                k_layer += 1
        grad_list = torch.cat(grad_list).cuda()
        _, indices = torch.topk(-1 * grad_list, int(len(grad_list) * ratio))  # 保留
        mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
        mask_flat_all_layer[indices] = 1.0
        count = 0
        percentage_mask_list = []
        k_layer = 0
        grad_abs_percentage_list = []
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients_length = len(parms.grad.abs().view(-1))
                mask_flat = mask_flat_all_layer[count:count + gradients_length].cuda()
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                count += gradients_length
                percentage_mask1 = mask_flat.sum().item() / float(gradients_length) * 100.0
                percentage_mask_list.append(percentage_mask1)
                grad_abs_percentage_list.append(grad_abs_sum_list[k_layer] / np.sum(grad_abs_sum_list))
                k_layer += 1

        model.zero_grad()

        return mask_grad_list

    def apply_grad_mask(self, model, mask_grad_list):
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)
